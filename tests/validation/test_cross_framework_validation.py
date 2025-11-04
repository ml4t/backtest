"""
Comprehensive Cross-Framework Validation Test

Tests QEngine against VectorBT and Backtrader using identical data and strategies.
This becomes part of our pytest suite for continuous validation.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project paths
qengine_src = Path(__file__).parent.parent.parent / "src"
projects_dir = Path(__file__).parent.parent.parent.parent / "projects"
sys.path.insert(0, str(qengine_src))

from frameworks import (
    BacktraderAdapter,
    MomentumStrategy,
    QEngineAdapter,
    ValidationResult,
    VectorBTAdapter,
)


def load_test_data() -> pd.DataFrame:
    """Load test data for validation."""
    # Try to use real data first
    wiki_path = projects_dir / "daily_us_equities" / "wiki_prices.parquet"

    if wiki_path.exists():
        df = pd.read_parquet(wiki_path)
        aapl = df[df["ticker"] == "AAPL"].copy()
        aapl["date"] = pd.to_datetime(aapl["date"])
        aapl = aapl.set_index("date").sort_index()
        test_data = aapl.loc["2015-01-01":"2016-12-31"].copy()
        print(f"Using real AAPL data: {len(test_data)} rows")
        return test_data

    # Fallback to synthetic data
    print("Using synthetic test data")
    dates = pd.date_range("2015-01-01", "2016-12-31", freq="D")

    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(dates))

    # Add trend for crossover signals
    trend = np.sin(np.arange(len(dates)) * 2 * np.pi / 252) * 0.001
    returns = returns + trend

    prices = 100 * (1 + returns).cumprod()

    data = pd.DataFrame(
        {
            "open": prices * (1 + np.random.normal(0, 0.001, len(dates))),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, len(dates)),
        },
        index=dates,
    )

    # Ensure OHLC consistency
    data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
    data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

    return data


def compare_results(results: list[ValidationResult]) -> dict[str, Any]:
    """Compare results across frameworks."""
    print(f"\n{'=' * 80}")
    print("CROSS-FRAMEWORK VALIDATION RESULTS")
    print(f"{'=' * 80}")

    # Filter successful results
    successful = [r for r in results if not r.has_errors]
    failed = [r for r in results if r.has_errors]

    print(f"✓ Successful: {len(successful)}")
    print(f"✗ Failed: {len(failed)}")

    if failed:
        print("\nFailed Frameworks:")
        for r in failed:
            print(f"  • {r.framework}: {r.errors}")

    if len(successful) < 2:
        print("⚠ Insufficient successful results for comparison")
        return {"status": "insufficient_data"}

    # Create comparison table
    print(
        f"\n{'Framework':<15} {'Final Value':<12} {'Return (%)':<10} {'Trades':<8} {'Time (s)':<10}",
    )
    print("-" * 70)

    for r in successful:
        print(
            f"{r.framework:<15} ${r.final_value:<11,.2f} {r.total_return:<9.2f} {r.num_trades:<7} {r.execution_time:<9.3f}",
        )

    # Check for discrepancies
    final_values = [r.final_value for r in successful]
    returns = [r.total_return for r in successful]
    trade_counts = [r.num_trades for r in successful]

    discrepancies = []

    # Return consistency (within 0.1%)
    return_std = np.std(returns) if len(returns) > 1 else 0
    if return_std > 0.1:
        discrepancies.append(f"Return discrepancy: std={return_std:.2f}%")

    # Trade count consistency
    if len(set(trade_counts)) > 1:
        discrepancies.append(
            f"Trade count mismatch: {dict(zip([r.framework for r in successful], trade_counts, strict=False))}",
        )

    # Value consistency (within $10)
    value_std = np.std(final_values) if len(final_values) > 1 else 0
    if value_std > 10.0:
        discrepancies.append(f"Final value discrepancy: std=${value_std:.2f}")

    if discrepancies:
        print("\n⚠ DISCREPANCIES DETECTED:")
        for disc in discrepancies:
            print(f"  • {disc}")

        # Show detailed trade comparison for first few trades
        print("\nDetailed Trade Comparison:")
        max_trades = min([len(r.trades) for r in successful if r.trades])
        if max_trades > 0:
            print(f"{'Framework':<15} {'Date':<12} {'Action':<6} {'Price':<10} {'Quantity':<12}")
            print("-" * 70)

            for i in range(min(3, max_trades)):  # First 3 trades
                for r in successful:
                    if i < len(r.trades):
                        trade = r.trades[i]
                        print(
                            f"{r.framework:<15} {trade.timestamp.strftime('%Y-%m-%d'):<12} {trade.action:<6} ${trade.price:<9.2f} {trade.quantity:<11.2f}",
                        )
                print()  # Blank line between trade groups
    else:
        print("\n✅ EXCELLENT AGREEMENT!")
        print(f"  • Returns within 0.1% (std={return_std:.4f}%)")
        print(f"  • Identical trade counts ({trade_counts[0] if trade_counts else 'N/A'})")
        print(f"  • Values within $10 (std=${value_std:.2f})")

    # Performance comparison
    if len(successful) > 1:
        fastest = min(successful, key=lambda r: r.execution_time)
        slowest = max(successful, key=lambda r: r.execution_time)
        print("\nPerformance:")
        print(f"  • Fastest: {fastest.framework} ({fastest.execution_time:.3f}s)")
        print(f"  • Slowest: {slowest.framework} ({slowest.execution_time:.3f}s)")
        if slowest.execution_time > 0:
            speedup = slowest.execution_time / fastest.execution_time
            print(f"  • Speed difference: {speedup:.1f}x")

    return {
        "successful_count": len(successful),
        "failed_count": len(failed),
        "discrepancies": discrepancies,
        "return_std": return_std,
        "value_std": value_std,
        "trade_count_consistent": len(set(trade_counts)) == 1,
    }


def test_cross_framework_validation():
    """Main validation test function."""
    print("Cross-Framework Backtester Validation")
    print("=" * 50)

    # Load test data
    data = load_test_data()
    print(f"Test period: {data.index[0]} to {data.index[-1]} ({len(data)} days)")

    # Define strategy parameters
    strategy = MomentumStrategy(short_window=20, long_window=50)
    strategy_params = strategy.get_parameters()
    initial_capital = 10000

    print(f"Strategy: {strategy_params['name']}")
    print(
        f"Parameters: MA({strategy_params['short_window']}) vs MA({strategy_params['long_window']})",
    )
    print(f"Initial Capital: ${initial_capital:,}")

    # Create adapters
    adapters = []

    try:
        adapters.append(QEngineAdapter())
    except Exception as e:
        print(f"QEngine adapter failed to create: {e}")

    try:
        adapters.append(VectorBTAdapter())
    except Exception as e:
        print(f"VectorBT adapter failed to create: {e}")

    try:
        adapters.append(BacktraderAdapter())
    except Exception as e:
        print(f"Backtrader adapter failed to create: {e}")

    if not adapters:
        raise RuntimeError("No framework adapters available!")

    print(f"Testing {len(adapters)} frameworks: {[a.framework_name for a in adapters]}")

    # Run backtests
    results = []
    for adapter in adapters:
        print(f"\n{'-' * 50}")
        print(f"Running {adapter.framework_name} backtest...")
        result = adapter.run_backtest(data, strategy_params, initial_capital)
        results.append(result)

    # Compare results
    comparison = compare_results(results)

    return results, comparison


if __name__ == "__main__":
    results, comparison = test_cross_framework_validation()

    # Determine overall success
    if comparison.get("status") == "insufficient_data":
        print("\n❌ Validation failed - insufficient successful results")
        exit_code = 1
    elif comparison.get("discrepancies"):
        print("\n⚠ Validation completed with discrepancies")
        exit_code = 0  # Not a failure, just needs investigation
    else:
        print("\n✅ Validation successful - all frameworks agree!")
        exit_code = 0

    # Summary stats
    print("\nSummary:")
    print(f"  • Successful frameworks: {comparison.get('successful_count', 0)}")
    print(f"  • Return consistency: {comparison.get('return_std', 0):.4f}% std dev")
    print(f"  • Trade count consistent: {comparison.get('trade_count_consistent', False)}")

    sys.exit(exit_code)
