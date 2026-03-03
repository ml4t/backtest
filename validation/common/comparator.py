"""Comparison utilities for validation results."""

from __future__ import annotations

from .types import CheckResult, ComparisonResult, FrameworkResult, ScenarioConfig, Tolerance


def compare_results(
    scenario: ScenarioConfig,
    framework_result: FrameworkResult,
    ml4t_result: FrameworkResult,
    tolerance: Tolerance | None = None,
) -> ComparisonResult:
    """Compare framework result against ml4t result.

    Args:
        scenario: Scenario configuration.
        framework_result: Result from external framework.
        ml4t_result: Result from ml4t.backtest.
        tolerance: Comparison tolerances (uses scenario defaults if None).

    Returns:
        ComparisonResult with individual check results.
    """
    if tolerance is None:
        if scenario.tolerances and framework_result.framework.lower().replace(" ", "_") in scenario.tolerances:
            tolerance = scenario.tolerances[framework_result.framework.lower().replace(" ", "_")]
        elif scenario.default_tolerance:
            tolerance = scenario.default_tolerance
        else:
            tolerance = Tolerance()

    checks: list[CheckResult] = []

    # Trade count
    trade_diff = abs(framework_result.num_trades - ml4t_result.num_trades)
    checks.append(CheckResult(
        name="trade_count",
        passed=trade_diff <= tolerance.trade_count,
        message=(
            f"{framework_result.framework}={framework_result.num_trades}, "
            f"ML4T={ml4t_result.num_trades}"
        ),
        expected=framework_result.num_trades,
        actual=ml4t_result.num_trades,
    ))

    # Final value
    fw_value = framework_result.final_value
    ml4t_value = ml4t_result.final_value
    value_diff = abs(fw_value - ml4t_value)
    value_pct = value_diff / abs(fw_value) * 100 if fw_value != 0 else 0
    checks.append(CheckResult(
        name="final_value",
        passed=value_pct < tolerance.value_pct,
        message=(
            f"{framework_result.framework}=${fw_value:,.2f}, "
            f"ML4T=${ml4t_value:,.2f} (diff={value_pct:.4f}%)"
        ),
        expected=fw_value,
        actual=ml4t_value,
    ))

    # Total P&L
    fw_pnl = framework_result.total_pnl
    ml4t_pnl = ml4t_result.total_pnl
    pnl_diff = abs(fw_pnl - ml4t_pnl)
    checks.append(CheckResult(
        name="total_pnl",
        passed=pnl_diff < tolerance.pnl_abs,
        message=(
            f"{framework_result.framework}=${fw_pnl:,.2f}, "
            f"ML4T=${ml4t_pnl:,.2f} (diff=${pnl_diff:.2f})"
        ),
        expected=fw_pnl,
        actual=ml4t_pnl,
    ))

    # Extra checks
    if "commission" in scenario.extra_checks:
        fw_comm = framework_result.extra.get("total_commission")
        ml4t_comm = ml4t_result.extra.get("total_commission")
        # Only compare if the framework provides commission data
        if fw_comm is not None and ml4t_comm is not None:
            comm_diff = abs(fw_comm - ml4t_comm)
            checks.append(CheckResult(
                name="total_commission",
                passed=comm_diff < tolerance.commission_abs,
                message=f"{framework_result.framework}=${fw_comm:.2f}, ML4T=${ml4t_comm:.2f} (diff=${comm_diff:.2f})",
                expected=fw_comm,
                actual=ml4t_comm,
            ))

    if "exit_price" in scenario.extra_checks:
        fw_exit = framework_result.extra.get("exit_price")
        ml4t_exit = ml4t_result.extra.get("exit_price")
        if fw_exit is not None and ml4t_exit is not None:
            exit_diff = abs(fw_exit - ml4t_exit)
            checks.append(CheckResult(
                name="exit_price",
                passed=exit_diff < tolerance.exit_price_abs,
                message=f"{framework_result.framework}=${fw_exit:.2f}, ML4T=${ml4t_exit:.2f} (diff=${exit_diff:.2f})",
                expected=fw_exit,
                actual=ml4t_exit,
            ))

    # Trade-by-trade comparison (if counts match)
    if framework_result.num_trades == ml4t_result.num_trades and framework_result.trades and ml4t_result.trades:
        trade_matches = 0
        for fw_t, ml4t_t in zip(framework_result.trades, ml4t_result.trades):
            entry_ok = abs(fw_t.get("entry_price", 0) - ml4t_t.get("entry_price", 0)) < tolerance.exit_price_abs
            exit_ok = abs(fw_t.get("exit_price", 0) - ml4t_t.get("exit_price", 0)) < tolerance.exit_price_abs
            if entry_ok and exit_ok:
                trade_matches += 1

        if framework_result.num_trades > 0:
            match_pct = trade_matches / framework_result.num_trades * 100
            checks.append(CheckResult(
                name="trade_level_match",
                passed=match_pct >= 99.0,
                message=f"{trade_matches}/{framework_result.num_trades} ({match_pct:.1f}%)",
            ))

    all_passed = all(c.passed for c in checks)

    return ComparisonResult(
        scenario_id=scenario.id,
        framework=framework_result.framework,
        passed=all_passed,
        checks=checks,
    )


def print_comparison(result: ComparisonResult, verbose: bool = True) -> None:
    """Print comparison result to stdout."""
    print(f"\n{'=' * 70}")
    print(f"COMPARISON: {result.framework} vs ml4t.backtest (Scenario {result.scenario_id})")
    print("=" * 70)

    for check in result.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"  {check.name}: {check.message} [{status}]")

    print(f"\n{'=' * 70}")
    if result.passed:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")
    print("=" * 70)
