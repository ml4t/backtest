#!/usr/bin/env python3
"""Unified scenario runner for the validation suite.

Usage:
    # Single scenario, single framework
    python validation/run_scenario.py --scenario 01 --framework backtrader

    # All scenarios for one framework
    python validation/run_scenario.py --framework vectorbt_oss

    # Full matrix (all scenarios, all frameworks)
    python validation/run_scenario.py --all

    # Dry run (list combinations without executing)
    python validation/run_scenario.py --dry-run

    # Verbose output with trade-level details
    python validation/run_scenario.py --scenario 01 --framework backtrader --verbose
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
VALIDATION_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(VALIDATION_DIR))

from common.comparator import compare_results, print_comparison  # noqa: E402
from common.framework_registry import load_framework_manifest  # noqa: E402
from common.ml4t_runner import run_ml4t  # noqa: E402
from common.provenance import build_record_provenance, generate_inputs  # noqa: E402
from common.types import (  # noqa: E402
    ComparisonResult,
    FrameworkResult,
    ValidationRecord,
    ValidationSkipped,
    ValidationStatus,
)
from scenarios.definitions import SCENARIOS  # noqa: E402

# Framework module map
FRAMEWORK_MODULES = {
    "vectorbt_pro": "frameworks.vectorbt_pro",
    "vectorbt_oss": "frameworks.vectorbt_oss",
    "backtrader": "frameworks.backtrader",
    "zipline": "frameworks.zipline",
}
FRAMEWORK_MANIFEST = load_framework_manifest()


def _record(
    scenario_id: str,
    framework: str,
    status: ValidationStatus,
    *,
    required: bool = True,
    detail: str | None = None,
    started_at: float | None = None,
    provenance: dict[str, Any] | None = None,
    framework_result: FrameworkResult | None = None,
    ml4t_result: FrameworkResult | None = None,
    comparison: ComparisonResult | None = None,
) -> ValidationRecord:
    scenario = SCENARIOS.get(scenario_id)
    return ValidationRecord(
        framework=framework,
        scenario_id=scenario_id,
        scenario_name=scenario.name if scenario else f"Scenario {scenario_id}",
        status=status,
        required=required,
        detail=detail,
        duration_seconds=time.perf_counter() - started_at if started_at is not None else None,
        provenance=provenance,
        framework_result=framework_result,
        ml4t_result=ml4t_result,
        comparison=comparison,
    )


def _malformed_result_detail(result: object) -> str | None:
    if not isinstance(result, FrameworkResult):
        return f"Adapter returned {type(result).__name__}, expected FrameworkResult"
    try:
        FrameworkResult.from_dict(result.to_dict())
    except (TypeError, ValueError) as error:
        return str(error)
    if result.num_trades != len(result.trades):
        return (
            "FrameworkResult.num_trades must equal the retained closed-trade count: "
            f"{result.num_trades} != {len(result.trades)}"
        )
    return None


def run_single(
    scenario_id: str,
    framework: str,
    verbose: bool = False,
) -> ValidationRecord:
    """Run a single scenario-framework pair.

    Returns:
        A terminal, machine-readable validation record.
    """
    started_at = time.perf_counter()
    scenario = SCENARIOS.get(scenario_id)
    if not scenario:
        print(f"Unknown scenario: {scenario_id}")
        return _record(
            scenario_id,
            framework,
            ValidationStatus.MISSING_SCENARIO,
            detail=f"Scenario {scenario_id} is not defined",
            started_at=started_at,
        )

    if framework not in scenario.supported_frameworks:
        print(f"Scenario {scenario_id} ({scenario.name}) does not support {framework}")
        return _record(
            scenario_id,
            framework,
            ValidationStatus.UNSUPPORTED,
            required=False,
            detail="Scenario explicitly excludes this framework",
            started_at=started_at,
        )

    print(f"\n{'=' * 70}")
    print(f"Scenario {scenario_id}: {scenario.name} ({framework})")
    print(f"{'=' * 70}")

    # Generate data
    print("\nGenerating test data...")
    try:
        prices_df, entries, exits = generate_inputs(scenario, framework)
    except Exception as error:
        print(f"   ERROR: data generation failed: {error}")
        return _record(
            scenario_id,
            framework,
            ValidationStatus.ADAPTER_FAILURE,
            detail=f"Data generation failed: {error}",
            started_at=started_at,
        )

    print(f"   Bars: {len(prices_df)}")
    print(f"   Entry signals: {entries.sum()}")
    if exits is not None:
        print(f"   Exit signals: {exits.sum()}")

    # Run external framework
    print(f"\nRunning {framework}...")
    try:
        fw_module = importlib.import_module(FRAMEWORK_MODULES[framework])
    except ImportError as error:
        print(f"   ERROR: adapter import failed: {error}")
        return _record(
            scenario_id,
            framework,
            ValidationStatus.ADAPTER_IMPORT_FAILURE,
            detail=str(error),
            started_at=started_at,
        )

    try:
        fw_result = fw_module.run(scenario, prices_df, entries, exits)
    except ValidationSkipped as error:
        print(f"   SKIP: {error}")
        return _record(
            scenario_id,
            framework,
            ValidationStatus.SKIPPED,
            detail=str(error),
            started_at=started_at,
        )
    except ImportError as error:
        print(f"   UNAVAILABLE: {error}")
        return _record(
            scenario_id,
            framework,
            ValidationStatus.UNAVAILABLE,
            detail=str(error),
            started_at=started_at,
        )
    except Exception as error:
        print(f"   ERROR: {error}")
        return _record(
            scenario_id,
            framework,
            ValidationStatus.ADAPTER_FAILURE,
            detail=str(error),
            started_at=started_at,
        )

    malformed_detail = _malformed_result_detail(fw_result)
    if malformed_detail:
        print(f"   ERROR: {malformed_detail}")
        return _record(
            scenario_id,
            framework,
            ValidationStatus.MALFORMED_OUTPUT,
            detail=malformed_detail,
            started_at=started_at,
        )

    try:
        print(f"   Trades: {fw_result.num_trades}")
        print(f"   Final Value: ${fw_result.final_value:,.2f}")
    except Exception as error:
        return _record(
            scenario_id,
            framework,
            ValidationStatus.MALFORMED_OUTPUT,
            detail=f"Could not render adapter result: {error}",
            started_at=started_at,
        )

    # Run ml4t
    print("\nRunning ml4t.backtest...")
    try:
        ml4t_result = run_ml4t(scenario, prices_df, entries, exits, framework=framework)
        print(f"   Trades: {ml4t_result.num_trades}")
        print(f"   Final Value: ${ml4t_result.final_value:,.2f}")
    except Exception as error:
        print(f"   ERROR: {error}")
        return _record(
            scenario_id,
            framework,
            ValidationStatus.ML4T_FAILURE,
            detail=str(error),
            started_at=started_at,
        )

    # Compare
    try:
        result = compare_results(scenario, fw_result, ml4t_result)
        print_comparison(result, verbose=verbose)
    except Exception as error:
        print(f"   ERROR: comparison failed: {error}")
        return _record(
            scenario_id,
            framework,
            ValidationStatus.MALFORMED_OUTPUT,
            detail=f"Could not compare validation results: {error}",
            started_at=started_at,
        )

    try:
        provenance = build_record_provenance(
            scenario=scenario,
            framework=framework,
            target=FRAMEWORK_MANIFEST.targets[framework],
            prices=prices_df,
            entries=entries,
            exits=exits,
            framework_result=fw_result,
            ml4t_result=ml4t_result,
        )
    except Exception as error:
        return _record(
            scenario_id,
            framework,
            ValidationStatus.MALFORMED_OUTPUT,
            detail=f"Could not retain validation provenance: {error}",
            started_at=started_at,
        )

    status = ValidationStatus.PASS if result.passed else ValidationStatus.COMPARISON_FAILURE
    return _record(
        scenario_id,
        framework,
        status,
        detail=None if result.passed else result.summary,
        started_at=started_at,
        provenance=provenance,
        framework_result=fw_result,
        ml4t_result=ml4t_result,
        comparison=result,
    )


def run_matrix(
    frameworks: list[str] | None = None,
    scenarios: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, dict[str, ValidationRecord]]:
    """Run all specified scenario-framework combinations.

    Returns:
        Nested dict of terminal records by framework and scenario.
    """
    if frameworks is None:
        frameworks = list(FRAMEWORK_MODULES.keys())
    if scenarios is None:
        scenarios = list(SCENARIOS.keys())

    results: dict[str, dict[str, ValidationRecord]] = {}

    for fw in frameworks:
        results[fw] = {}
        for sid in scenarios:
            try:
                results[fw][sid] = run_single(sid, fw, verbose=verbose)
            except Exception as error:
                print(f"\nERROR in {fw}/{sid}: {error}")
                results[fw][sid] = _record(
                    sid,
                    fw,
                    ValidationStatus.ADAPTER_FAILURE,
                    detail=f"Unhandled validation error: {error}",
                )

    return results


def _flatten(results: dict[str, dict[str, ValidationRecord]]) -> list[ValidationRecord]:
    return [record for scenarios in results.values() for record in scenarios.values()]


def _release_gate_passed(records: list[ValidationRecord]) -> bool:
    return any(record.required for record in records) and not any(
        record.release_blocking for record in records
    )


def print_summary(results: dict[str, dict[str, ValidationRecord]]) -> None:
    """Print summary table of all results."""
    print(f"\n{'=' * 70}")
    print("VALIDATION SUMMARY")
    print("=" * 70)

    counts = dict.fromkeys(ValidationStatus, 0)

    print(f"\n{'Framework':<20} {'Scenario':<30} {'Status'}")
    print("-" * 60)

    for fw, scenarios in results.items():
        for sid, record in scenarios.items():
            counts[record.status] += 1
            print(f"  {fw:<18} {sid}: {record.scenario_name:<24} {record.status.value.upper()}")

    nonzero = [f"{status.value}={count}" for status, count in counts.items() if count]
    print(f"\nTotal: {', '.join(nonzero)}")
    print("=" * 70)


def _write_result_json(path: Path, records: list[ValidationRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(records) == 1:
        payload: object = records[0].to_dict()
    else:
        payload = {
            "schema_version": 1,
            "records": [record.to_dict() for record in records],
        }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dry_run(
    frameworks: list[str] | None = None,
    scenarios: list[str] | None = None,
) -> None:
    """List all scenario-framework combinations without executing."""
    if frameworks is None:
        frameworks = list(FRAMEWORK_MODULES.keys())
    if scenarios is None:
        scenarios = list(SCENARIOS.keys())

    print("Validation Matrix (dry run)")
    print("=" * 70)

    count = 0
    for fw in frameworks:
        for sid in scenarios:
            scenario = SCENARIOS.get(sid)
            if not scenario:
                continue
            supported = fw in scenario.supported_frameworks
            status = "RUN" if supported else "SKIP (unsupported)"
            print(f"  {fw:<18} {sid}: {scenario.name:<24} {status}")
            if supported:
                count += 1

    print(f"\nTotal combinations to run: {count}")


def main():
    parser = argparse.ArgumentParser(description="Run validation scenarios")
    parser.add_argument("--scenario", type=str, help="Scenario ID (e.g., 01, 02, ...)")
    parser.add_argument("--framework", choices=tuple(FRAMEWORK_MODULES), help="Framework name")
    parser.add_argument("--all", action="store_true", help="Run full matrix")
    parser.add_argument("--dry-run", action="store_true", help="List combinations only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--result-json",
        type=Path,
        help="Write machine-readable terminal record(s) to this path",
    )

    args = parser.parse_args()

    if args.dry_run:
        frameworks = [args.framework] if args.framework else None
        scenarios = [args.scenario] if args.scenario else None
        dry_run(frameworks, scenarios)
        return 0

    if args.all:
        results = run_matrix(verbose=args.verbose)
        print_summary(results)
        records = _flatten(results)
        if args.result_json:
            _write_result_json(args.result_json, records)
        return 0 if _release_gate_passed(records) else 1

    if args.scenario and args.framework:
        record = run_single(args.scenario, args.framework, verbose=args.verbose)
        if args.result_json:
            _write_result_json(args.result_json, [record])
        return 0 if _release_gate_passed([record]) else 1

    if args.framework:
        results = run_matrix(
            frameworks=[args.framework],
            verbose=args.verbose,
        )
        print_summary(results)
        records = _flatten(results)
        if args.result_json:
            _write_result_json(args.result_json, records)
        return 0 if _release_gate_passed(records) else 1

    if args.scenario:
        # Run scenario against all frameworks
        results = run_matrix(
            scenarios=[args.scenario],
            verbose=args.verbose,
        )
        print_summary(results)
        records = _flatten(results)
        if args.result_json:
            _write_result_json(args.result_json, records)
        return 0 if _release_gate_passed(records) else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
