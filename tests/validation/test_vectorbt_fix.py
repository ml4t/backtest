"""
Test script to verify VectorBT implementation fixes.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project paths
ml4t.backtest_src = Path(__file__).parent.parent.parent / "src"
validation_dir = Path(__file__).parent
projects_dir = Path(__file__).parent.parent.parent.parent / "projects"
sys.path.insert(0, str(ml4t.backtest_src))
sys.path.insert(0, str(validation_dir))

from frameworks.vectorbt_adapter import VectorBTAdapter


def create_test_data() -> pd.DataFrame:
    """Create synthetic test data."""
    dates = pd.date_range("2015-01-01", "2016-12-31", freq="D")

    # Create realistic price series with trend
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(dates))

    # Add some trend for crossover signals
    trend = np.sin(np.arange(len(dates)) * 2 * np.pi / 252) * 0.001
    returns = returns + trend

    prices = 100 * (1 + returns).cumprod()

    # Create OHLCV data
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

    # Ensure high >= close >= low and high >= open >= low
    data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
    data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

    return data


def test_vectorbt_implementation():
    """Test VectorBT adapter implementation."""
    print("Testing VectorBT Implementation Fix")
    print("=" * 50)

    # Create test data
    test_data = create_test_data()
    print(
        f"Created test data: {len(test_data)} rows from {test_data.index[0]} to {test_data.index[-1]}",
    )

    # Set up strategy parameters
    strategy_params = {"name": "MovingAverageCrossover", "short_window": 20, "long_window": 50}

    # Create adapter
    adapter = VectorBTAdapter()

    # Run backtest
    print(f"\nRunning {adapter.framework_name} backtest...")
    result = adapter.run_backtest(test_data, strategy_params, initial_capital=10000)

    # Display results
    print(f"\n{adapter.framework_name} Results:")
    print(f"  Status: {'✓ Success' if not result.has_errors else '✗ Failed'}")

    if result.has_errors:
        print("  Errors:")
        for error in result.errors:
            print(f"    • {error}")
    else:
        print(f"  Final Value: ${result.final_value:,.2f}")
        print(f"  Total Return: {result.total_return:.2f}%")
        print(f"  Number of Trades: {result.num_trades}")
        print(f"  Win Rate: {result.win_rate:.1%}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"  Execution Time: {result.execution_time:.3f}s")
        print(f"  Memory Usage: {result.memory_usage:.1f}MB")

        if result.trades:
            print("  First 3 trades:")
            for i, trade in enumerate(result.trades[:3]):
                print(f"    {i + 1}. {trade}")

    # Assert the backtest succeeded
    assert not result.has_errors, f"Backtest failed with errors: {result.errors}"
    assert result.num_trades > 0, "No trades generated"
    assert result.final_value > 0, "Final value must be positive"


if __name__ == "__main__":
    try:
        test_vectorbt_implementation()
        print("\n✅ VectorBT implementation fixed successfully!")
    except AssertionError as e:
        print(f"\n❌ VectorBT implementation still has issues: {e}")
        sys.exit(1)
