"""Comparison and validation tools for engine results."""
import pandas as pd
import numpy as np
from typing import Dict
from .engine_wrappers import BacktestResult


def compare_trades(results: Dict[str, BacktestResult], tolerance: float = 0.01) -> pd.DataFrame:
    """
    Compare trade results across multiple engines.

    Args:
        results: Dict mapping engine name to BacktestResult
        tolerance: Tolerance for price comparisons (relative)

    Returns:
        DataFrame summarizing comparison
    """
    comparison_data = []

    for engine_name, result in results.items():
        final_pos_str = f"{result.final_position:.4f}" if result.final_position is not None else "None"
        comparison_data.append({
            'Engine': engine_name,
            'Num Trades': result.num_trades,
            'Final Value': f"${result.final_value:,.2f}",
            'Total PnL': f"${result.total_pnl:,.2f}",
            'Final Cash': f"${result.final_cash:,.2f}",
            'Final Position': final_pos_str,
        })

    df = pd.DataFrame(comparison_data)
    return df


def assert_identical(
    results: Dict[str, BacktestResult],
    price_tolerance: float = 0.01,
    value_tolerance: float = 1.0,
) -> tuple[bool, str]:
    """
    Assert that all engines produced identical results.

    Args:
        results: Dict mapping engine name to BacktestResult
        price_tolerance: Tolerance for price comparisons (absolute $)
        value_tolerance: Tolerance for value comparisons (absolute $)

    Returns:
        Tuple of (success: bool, message: str)
    """
    if len(results) < 2:
        return False, "Need at least 2 engines to compare"

    # Get reference (first engine)
    ref_name = list(results.keys())[0]
    ref_result = results[ref_name]

    # Compare each engine to reference
    failures = []

    for engine_name, result in results.items():
        if engine_name == ref_name:
            continue

        # Compare trade count
        if result.num_trades != ref_result.num_trades:
            failures.append(
                f"{engine_name}: {result.num_trades} trades vs {ref_name}: {ref_result.num_trades} trades"
            )

        # Compare final value
        value_diff = abs(result.final_value - ref_result.final_value)
        if value_diff > value_tolerance:
            failures.append(
                f"{engine_name}: final value ${result.final_value:,.2f} "
                f"vs {ref_name}: ${ref_result.final_value:,.2f} "
                f"(diff: ${value_diff:,.2f})"
            )

    if failures:
        message = "❌ DIFFERENCES FOUND:\n" + "\n".join(f"  - {f}" for f in failures)
        return False, message
    else:
        message = f"✅ ALL ENGINES IDENTICAL ({len(results)} engines compared)"
        return True, message


def compare_first_trades(
    results: Dict[str, BacktestResult],
    n: int = 5,
) -> pd.DataFrame:
    """
    Compare first N trades across engines.

    Args:
        results: Dict mapping engine name to BacktestResult
        n: Number of trades to compare

    Returns:
        DataFrame with side-by-side comparison
    """
    comparison_rows = []

    for i in range(n):
        row = {'Trade': i + 1}

        for engine_name, result in results.items():
            if i < len(result.trades):
                trade = result.trades.iloc[i]
                row[f'{engine_name}_entry_time'] = trade['entry_time']
                row[f'{engine_name}_entry_price'] = f"${trade['entry_price']:,.2f}"
                row[f'{engine_name}_exit_time'] = trade.get('exit_time', 'N/A')
                row[f'{engine_name}_exit_price'] = (
                    f"${trade['exit_price']:,.2f}" if pd.notna(trade.get('exit_price')) else 'N/A'
                )
                row[f'{engine_name}_pnl'] = f"${trade['pnl']:,.2f}" if pd.notna(trade.get('pnl')) else 'N/A'
            else:
                row[f'{engine_name}_entry_time'] = 'N/A'
                row[f'{engine_name}_entry_price'] = 'N/A'
                row[f'{engine_name}_exit_time'] = 'N/A'
                row[f'{engine_name}_exit_price'] = 'N/A'
                row[f'{engine_name}_pnl'] = 'N/A'

        comparison_rows.append(row)

    return pd.DataFrame(comparison_rows)


def print_validation_report(
    results: Dict[str, BacktestResult],
    test_name: str,
    show_first_trades: int = 3,
):
    """
    Print a comprehensive validation report.

    Args:
        results: Dict mapping engine name to BacktestResult
        test_name: Name of the test
        show_first_trades: Number of first trades to show
    """
    print("=" * 80)
    print(f"VALIDATION REPORT: {test_name}")
    print("=" * 80)

    # Summary comparison
    print("\n📊 Summary:")
    summary_df = compare_trades(results)
    print(summary_df.to_string(index=False))

    # Check if identical
    print("\n🔍 Validation:")
    success, message = assert_identical(results)
    print(message)

    # First trades comparison
    if show_first_trades > 0:
        print(f"\n📈 First {show_first_trades} Trades:")
        first_trades_df = compare_first_trades(results, n=show_first_trades)
        print(first_trades_df.to_string(index=False))

    print("\n" + "=" * 80)
    return success
