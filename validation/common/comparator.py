"""Comparison utilities for validation results."""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal

from .types import CheckResult, ComparisonResult, FrameworkResult, ScenarioConfig, Tolerance

CANONICAL_QUANTUM = Decimal("0.00000001")


def _canonical_number(value: int | float) -> Decimal:
    """Convert binary framework output to the shared eight-decimal fixed-point domain."""
    return Decimal(str(value)).quantize(CANONICAL_QUANTUM, rounding=ROUND_HALF_EVEN)


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
        tolerance: Diagnostic thresholds reported with exact release checks.

    Returns:
        ComparisonResult with individual check results.
    """
    if tolerance is None:
        framework_key = framework_result.framework.lower().replace(" ", "_")
        if scenario.tolerances and framework_key in scenario.tolerances:
            tolerance = scenario.tolerances[framework_key]
        elif scenario.default_tolerance:
            tolerance = scenario.default_tolerance
        else:
            tolerance = Tolerance()

    checks: list[CheckResult] = []

    # Trade count
    trade_diff = abs(framework_result.num_trades - ml4t_result.num_trades)
    checks.append(
        CheckResult(
            name="trade_count",
            passed=trade_diff == 0,
            message=(
                f"{framework_result.framework}={framework_result.num_trades}, "
                f"ML4T={ml4t_result.num_trades}, exact_diff={trade_diff}; "
                f"diagnostic_limit={tolerance.trade_count}"
            ),
            expected=framework_result.num_trades,
            actual=ml4t_result.num_trades,
        )
    )

    # Final value
    fw_value = framework_result.final_value
    ml4t_value = ml4t_result.final_value
    value_diff = abs(fw_value - ml4t_value)
    value_pct = value_diff / abs(fw_value) * 100 if fw_value != 0 else 0
    checks.append(
        CheckResult(
            name="final_value",
            passed=_canonical_number(fw_value) == _canonical_number(ml4t_value),
            message=(
                f"{framework_result.framework}=${fw_value:,.10f}, "
                f"ML4T=${ml4t_value:,.10f} (exact_diff=${value_diff:.10f}, "
                f"{value_pct:.10f}%; diagnostic_limit={tolerance.value_pct}%)"
            ),
            expected=fw_value,
            actual=ml4t_value,
        )
    )

    # Total P&L
    fw_pnl = framework_result.total_pnl
    ml4t_pnl = ml4t_result.total_pnl
    pnl_diff = abs(fw_pnl - ml4t_pnl)
    checks.append(
        CheckResult(
            name="total_pnl",
            passed=_canonical_number(fw_pnl) == _canonical_number(ml4t_pnl),
            message=(
                f"{framework_result.framework}=${fw_pnl:,.10f}, "
                f"ML4T=${ml4t_pnl:,.10f} (exact_diff=${pnl_diff:.10f}; "
                f"diagnostic_limit=${tolerance.pnl_abs})"
            ),
            expected=fw_pnl,
            actual=ml4t_pnl,
        )
    )

    # Extra checks
    if "commission" in scenario.extra_checks:
        fw_comm = framework_result.extra.get("total_commission")
        ml4t_comm = ml4t_result.extra.get("total_commission")
        if fw_comm is None or ml4t_comm is None:
            checks.append(
                CheckResult(
                    name="total_commission",
                    passed=False,
                    message="Required commission output is missing",
                    expected=fw_comm,
                    actual=ml4t_comm,
                )
            )
        else:
            comm_diff = abs(fw_comm - ml4t_comm)
            checks.append(
                CheckResult(
                    name="total_commission",
                    passed=_canonical_number(fw_comm) == _canonical_number(ml4t_comm),
                    message=(
                        f"{framework_result.framework}=${fw_comm:.10f}, "
                        f"ML4T=${ml4t_comm:.10f} (exact_diff=${comm_diff:.10f}; "
                        f"diagnostic_limit=${tolerance.commission_abs})"
                    ),
                    expected=fw_comm,
                    actual=ml4t_comm,
                )
            )

    if "exit_price" in scenario.extra_checks:
        fw_exit = framework_result.extra.get("exit_price")
        ml4t_exit = ml4t_result.extra.get("exit_price")
        if fw_exit is None or ml4t_exit is None:
            checks.append(
                CheckResult(
                    name="exit_price",
                    passed=False,
                    message="Required exit-price output is missing",
                    expected=fw_exit,
                    actual=ml4t_exit,
                )
            )
        else:
            exit_diff = abs(fw_exit - ml4t_exit)
            checks.append(
                CheckResult(
                    name="exit_price",
                    passed=_canonical_number(fw_exit) == _canonical_number(ml4t_exit),
                    message=(
                        f"{framework_result.framework}=${fw_exit:.10f}, "
                        f"ML4T=${ml4t_exit:.10f} (exact_diff=${exit_diff:.10f}; "
                        f"diagnostic_limit=${tolerance.exit_price_abs})"
                    ),
                    expected=fw_exit,
                    actual=ml4t_exit,
                )
            )

    # Exact trade-by-trade release surface. Timestamps are excluded until every adapter exposes
    # them consistently; prices, size, direction, and PnL are mandatory for every reported trade.
    required_trade_fields = ("entry_price", "exit_price", "pnl", "size", "direction")
    trade_detail = "exact match"
    trade_level_passed = len(framework_result.trades) == len(ml4t_result.trades)
    if trade_level_passed:
        for index, (fw_trade, ml4t_trade) in enumerate(
            zip(framework_result.trades, ml4t_result.trades, strict=True)
        ):
            for field in required_trade_fields:
                if field not in fw_trade or field not in ml4t_trade:
                    trade_level_passed = False
                    trade_detail = f"trade {index} missing required field {field}"
                    break
                expected = fw_trade[field]
                actual = ml4t_trade[field]
                if field == "direction":
                    expected = str(expected).lower()
                    actual = str(actual).lower()
                else:
                    expected = _canonical_number(expected)
                    actual = _canonical_number(actual)
                if expected != actual:
                    trade_level_passed = False
                    trade_detail = (
                        f"trade {index} field {field}: expected={expected!r}, actual={actual!r}"
                    )
                    break
            if not trade_level_passed:
                break
    else:
        trade_detail = (
            f"record_count expected={len(framework_result.trades)}, "
            f"actual={len(ml4t_result.trades)}"
        )
    checks.append(
        CheckResult(
            name="trade_level_match",
            passed=trade_level_passed,
            message=trade_detail,
            expected=framework_result.trades,
            actual=ml4t_result.trades,
        )
    )

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
        if not verbose and check.passed:
            continue
        status = "PASS" if check.passed else "FAIL"
        print(f"  {check.name}: {check.message} [{status}]")

    print(f"\n{'=' * 70}")
    if result.passed:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")
    print("=" * 70)
