"""Comparison utilities for validation results."""

from __future__ import annotations

from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any

from .types import CheckResult, ComparisonResult, FrameworkResult, ScenarioConfig, Tolerance

CANONICAL_QUANTUM = Decimal("0.00000001")
CANONICAL_QUANTUM_TEXT = format(CANONICAL_QUANTUM, "f")
TRADE_FIELDS = (
    "entry_time",
    "exit_time",
    "asset",
    "direction",
    "size",
    "entry_price",
    "exit_price",
    "pnl",
    "commission",
)
FILL_FIELDS = ("timestamp", "asset", "side", "quantity", "price", "commission")
_TIMESTAMP_FIELDS = {"entry_time", "exit_time", "timestamp"}
_CASE_INSENSITIVE_FIELDS = {"direction", "side"}
_STRING_FIELDS = {"asset", *_CASE_INSENSITIVE_FIELDS}


def _canonical_number(value: int | float) -> Decimal:
    """Convert binary framework output to the shared eight-decimal fixed-point domain."""
    return Decimal(str(value)).quantize(CANONICAL_QUANTUM, rounding=ROUND_HALF_EVEN)


def _canonical_session(value: object) -> str:
    """Normalize a daily event timestamp to its ISO session date."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        session = value[:10]
        date.fromisoformat(session)
        return session
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        session = str(isoformat())[:10]
        date.fromisoformat(session)
        return session
    raise TypeError(f"unsupported timestamp type {type(value).__name__}")


def _canonical_field(field: str, value: Any) -> Decimal | str:
    if field in _TIMESTAMP_FIELDS:
        return _canonical_session(value)
    if field in _STRING_FIELDS:
        if not isinstance(value, str) or not value:
            raise TypeError("must be a nonempty string")
        return value.lower() if field in _CASE_INSENSITIVE_FIELDS else value
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("must be numeric")
    return _canonical_number(value)


def _compare_records(
    *,
    name: str,
    expected_records: list[dict[str, Any]],
    actual_records: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> CheckResult:
    detail = "exact ordered match"
    passed = len(expected_records) == len(actual_records)
    if not passed:
        detail = f"record_count expected={len(expected_records)}, actual={len(actual_records)}"
    else:
        for index, (expected_record, actual_record) in enumerate(
            zip(expected_records, actual_records, strict=True)
        ):
            for field in fields:
                if field not in expected_record or field not in actual_record:
                    passed = False
                    detail = f"record {index} missing required field {field}"
                    break
                try:
                    expected = _canonical_field(field, expected_record[field])
                    actual = _canonical_field(field, actual_record[field])
                except (ArithmeticError, TypeError, ValueError) as error:
                    passed = False
                    detail = f"record {index} field {field} is malformed: {error}"
                    break
                if expected != actual:
                    passed = False
                    detail = (
                        f"record {index} field {field}: expected={expected!r}, actual={actual!r}"
                    )
                    break
            if not passed:
                break
    return CheckResult(
        name=name,
        passed=passed,
        message=detail,
        expected=expected_records,
        actual=actual_records,
        difference=None if passed else detail,
        canonical_quantum=CANONICAL_QUANTUM_TEXT,
        diagnostic_limit=None,
    )


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
            difference=trade_diff,
            canonical_quantum="1",
            diagnostic_limit=tolerance.trade_count,
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
            difference=value_diff,
            canonical_quantum=CANONICAL_QUANTUM_TEXT,
            diagnostic_limit=tolerance.value_pct,
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
            difference=pnl_diff,
            canonical_quantum=CANONICAL_QUANTUM_TEXT,
            diagnostic_limit=tolerance.pnl_abs,
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
                    difference="missing",
                    canonical_quantum=CANONICAL_QUANTUM_TEXT,
                    diagnostic_limit=tolerance.commission_abs,
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
                    difference=comm_diff,
                    canonical_quantum=CANONICAL_QUANTUM_TEXT,
                    diagnostic_limit=tolerance.commission_abs,
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
                    difference="missing",
                    canonical_quantum=CANONICAL_QUANTUM_TEXT,
                    diagnostic_limit=tolerance.exit_price_abs,
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
                    difference=exit_diff,
                    canonical_quantum=CANONICAL_QUANTUM_TEXT,
                    diagnostic_limit=tolerance.exit_price_abs,
                )
            )

    checks.append(
        _compare_records(
            name="trade_level_match",
            expected_records=framework_result.trades,
            actual_records=ml4t_result.trades,
            fields=TRADE_FIELDS,
        )
    )
    checks.append(
        _compare_records(
            name="fill_level_match",
            expected_records=framework_result.fills,
            actual_records=ml4t_result.fills,
            fields=FILL_FIELDS,
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
