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
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
VALIDATION_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(VALIDATION_DIR))

from common import data_generators
from common.comparator import compare_results, print_comparison
from common.ml4t_runner import run_ml4t
from common.types import FrameworkResult
from scenarios.definitions import SCENARIOS

# Framework module map
FRAMEWORK_MODULES = {
    "vectorbt_pro": "frameworks.vectorbt_pro",
    "vectorbt_oss": "frameworks.vectorbt_oss",
    "backtrader": "frameworks.backtrader",
    "zipline": "frameworks.zipline",
}


def run_single(scenario_id: str, framework: str, verbose: bool = False) -> bool:
    """Run a single scenario-framework pair.

    Returns:
        True if validation passed.
    """
    scenario = SCENARIOS.get(scenario_id)
    if not scenario:
        print(f"Unknown scenario: {scenario_id}")
        return False

    if framework not in scenario.supported_frameworks:
        print(f"Scenario {scenario_id} ({scenario.name}) does not support {framework}")
        return True  # Not a failure, just skipped

    print(f"\n{'=' * 70}")
    print(f"Scenario {scenario_id}: {scenario.name} ({framework})")
    print(f"{'=' * 70}")

    # Generate data
    print("\nGenerating test data...")
    gen_func = getattr(data_generators, scenario.data_generator)
    data_result = gen_func(**scenario.data_kwargs)

    if len(data_result) == 3:
        prices_df, entries, exits = data_result
    else:
        prices_df, entries = data_result
        exits = None

    # Align to NYSE calendar for Zipline (which only operates on NYSE sessions)
    if framework == "zipline":
        import exchange_calendars as xcals

        nyse = xcals.get_calendar("XNYS")
        start_ts = prices_df.index[0]
        end_ts = prices_df.index[-1]
        if start_ts.tz is not None:
            start_ts = start_ts.tz_convert(None)
            end_ts = end_ts.tz_convert(None)
        sessions = nyse.sessions_in_range(start_ts, end_ts)
        naive_idx = prices_df.index.tz_localize(None) if prices_df.index.tz else prices_df.index
        valid_mask = naive_idx.isin(sessions)
        prices_df = prices_df[valid_mask].copy()
        entries = entries[valid_mask]
        if exits is not None:
            exits = exits[valid_mask]

    print(f"   Bars: {len(prices_df)}")
    print(f"   Entry signals: {entries.sum()}")
    if exits is not None:
        print(f"   Exit signals: {exits.sum()}")

    # Run external framework
    print(f"\nRunning {framework}...")
    try:
        fw_module = importlib.import_module(FRAMEWORK_MODULES[framework])
        fw_result = fw_module.run(scenario, prices_df, entries, exits)
        print(f"   Trades: {fw_result.num_trades}")
        print(f"   Final Value: ${fw_result.final_value:,.2f}")
    except ImportError as e:
        print(f"   SKIP: {e}")
        return True
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Run ml4t
    print("\nRunning ml4t.backtest...")
    try:
        ml4t_result = run_ml4t(scenario, prices_df, entries, exits, framework=framework)
        print(f"   Trades: {ml4t_result.num_trades}")
        print(f"   Final Value: ${ml4t_result.final_value:,.2f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Compare
    result = compare_results(scenario, fw_result, ml4t_result)
    print_comparison(result, verbose=verbose)

    return result.passed


def run_matrix(
    frameworks: list[str] | None = None,
    scenarios: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, dict[str, bool | None]]:
    """Run all specified scenario-framework combinations.

    Returns:
        Nested dict of {framework: {scenario_id: passed}}.
    """
    if frameworks is None:
        frameworks = list(FRAMEWORK_MODULES.keys())
    if scenarios is None:
        scenarios = list(SCENARIOS.keys())

    results: dict[str, dict[str, bool | None]] = {}

    for fw in frameworks:
        results[fw] = {}
        for sid in scenarios:
            scenario = SCENARIOS.get(sid)
            if not scenario:
                results[fw][sid] = None
                continue
            if fw not in scenario.supported_frameworks:
                results[fw][sid] = None
                continue

            try:
                passed = run_single(sid, fw, verbose=verbose)
                results[fw][sid] = passed
            except Exception as e:
                print(f"\nERROR in {fw}/{sid}: {e}")
                results[fw][sid] = False

    return results


def print_summary(results: dict[str, dict[str, bool | None]]) -> None:
    """Print summary table of all results."""
    print(f"\n{'=' * 70}")
    print("VALIDATION SUMMARY")
    print("=" * 70)

    total_pass = 0
    total_fail = 0
    total_skip = 0

    print(f"\n{'Framework':<20} {'Scenario':<30} {'Status'}")
    print("-" * 60)

    for fw, scenarios in results.items():
        for sid, passed in scenarios.items():
            scenario = SCENARIOS.get(sid)
            name = scenario.name if scenario else f"Scenario {sid}"

            if passed is True:
                status = "PASS"
                total_pass += 1
            elif passed is False:
                status = "FAIL"
                total_fail += 1
            else:
                status = "SKIP"
                total_skip += 1

            print(f"  {fw:<18} {sid}: {name:<24} {status}")

    print(f"\nTotal: {total_pass} passed, {total_fail} failed, {total_skip} skipped")
    print("=" * 70)


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
    parser.add_argument("--framework", type=str, help="Framework name")
    parser.add_argument("--all", action="store_true", help="Run full matrix")
    parser.add_argument("--dry-run", action="store_true", help="List combinations only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.dry_run:
        frameworks = [args.framework] if args.framework else None
        scenarios = [args.scenario] if args.scenario else None
        dry_run(frameworks, scenarios)
        return 0

    if args.all:
        results = run_matrix(verbose=args.verbose)
        print_summary(results)
        has_failures = any(
            v is False
            for scenarios in results.values()
            for v in scenarios.values()
        )
        return 1 if has_failures else 0

    if args.scenario and args.framework:
        passed = run_single(args.scenario, args.framework, verbose=args.verbose)
        return 0 if passed else 1

    if args.framework:
        results = run_matrix(
            frameworks=[args.framework],
            verbose=args.verbose,
        )
        print_summary(results)
        has_failures = any(
            v is False
            for scenarios in results.values()
            for v in scenarios.values()
        )
        return 1 if has_failures else 0

    if args.scenario:
        # Run scenario against all frameworks
        results = run_matrix(
            scenarios=[args.scenario],
            verbose=args.verbose,
        )
        print_summary(results)
        has_failures = any(
            v is False
            for scenarios in results.values()
            for v in scenarios.values()
        )
        return 1 if has_failures else 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
