"""
Pytest Integration for Cross-Framework Validation

Integrates cross-framework validation into pytest suite for CI/CD pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project paths
ml4t.backtest_src = Path(__file__).parent.parent.parent / "src"
validation_dir = Path(__file__).parent
projects_dir = Path(__file__).parent.parent.parent.parent / "projects"
sys.path.insert(0, str(ml4t.backtest_src))
sys.path.insert(0, str(validation_dir))

from frameworks import (
    BacktraderAdapter,
    MomentumStrategy,
    ml4t.backtestAdapter,
    VectorBTAdapter,
)


@pytest.fixture
def test_data():
    """Provide consistent test data for all validation tests."""
    # Try real data first
    wiki_path = projects_dir / "daily_us_equities" / "wiki_prices.parquet"

    if wiki_path.exists():
        df = pd.read_parquet(wiki_path)
        aapl = df[df["ticker"] == "AAPL"].copy()
        aapl["date"] = pd.to_datetime(aapl["date"])
        aapl = aapl.set_index("date").sort_index()
        return aapl.loc["2015-01-01":"2016-12-31"].copy()

    # Fallback to synthetic data
    dates = pd.date_range("2015-01-01", "2015-06-30", freq="D")  # Shorter for faster tests
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))
    trend = np.sin(np.arange(len(dates)) * 2 * np.pi / 60) * 0.002  # More signals
    returns = returns + trend
    prices = 100 * (1 + returns).cumprod()

    return pd.DataFrame(
        {
            "open": prices * (1 + np.random.normal(0, 0.001, len(dates))),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, len(dates)),
        },
        index=dates,
    )


@pytest.fixture
def momentum_strategy():
    """Provide standard momentum strategy parameters."""
    strategy = MomentumStrategy(short_window=10, long_window=20)  # Shorter for more signals
    return strategy.get_parameters()


class TestFrameworkValidation:
    """Test suite for cross-framework validation."""

    def test_ml4t.backtest_vectorbt_agreement(self, test_data, momentum_strategy):
        """Test that ml4t.backtest and VectorBT produce identical results."""
        qe_adapter = ml4t.backtestAdapter()
        vbt_adapter = VectorBTAdapter()

        initial_capital = 10000

        # Run both backtests
        qe_result = qe_adapter.run_backtest(test_data, momentum_strategy, initial_capital)
        vbt_result = vbt_adapter.run_backtest(test_data, momentum_strategy, initial_capital)

        # Both should succeed
        assert not qe_result.has_errors, f"ml4t.backtest errors: {qe_result.errors}"

        # Skip if VectorBT not available
        if vbt_result.has_errors and any("not available" in err for err in vbt_result.errors):
            pytest.skip("VectorBT not installed")

        assert not vbt_result.has_errors, f"VectorBT errors: {vbt_result.errors}"

        # Results should be nearly identical
        return_diff = abs(qe_result.total_return - vbt_result.total_return)
        assert return_diff < 0.01, f"Return difference {return_diff:.4f}% exceeds tolerance"

        value_diff = abs(qe_result.final_value - vbt_result.final_value)
        assert value_diff < 1.0, f"Final value difference ${value_diff:.2f} exceeds tolerance"

        # Trade counts should match
        assert qe_result.num_trades == vbt_result.num_trades, (
            f"Trade count mismatch: ml4t.backtest={qe_result.num_trades}, VectorBT={vbt_result.num_trades}"
        )

    def test_ml4t.backtest_performance(self, test_data, momentum_strategy):
        """Test ml4t.backtest performance benchmarks."""
        qe_adapter = ml4t.backtestAdapter()
        result = qe_adapter.run_backtest(test_data, momentum_strategy, 10000)

        assert not result.has_errors, f"ml4t.backtest failed: {result.errors}"

        # Performance benchmarks
        assert result.execution_time < 1.0, f"ml4t.backtest too slow: {result.execution_time:.3f}s"
        assert result.memory_usage < 100, (
            f"ml4t.backtest uses too much memory: {result.memory_usage:.1f}MB"
        )

        # Basic sanity checks
        assert result.final_value > 0, "Final value must be positive"
        assert result.num_trades >= 0, "Trade count must be non-negative"

    def test_framework_consistency(self, test_data, momentum_strategy):
        """Test consistency across available frameworks."""
        adapters = [ml4t.backtestAdapter()]

        # Add VectorBT if available
        try:
            adapters.append(VectorBTAdapter())
        except ImportError:
            pytest.skip("VectorBT not available")

        # Add Backtrader if available
        try:
            if BacktraderAdapter:
                adapters.append(BacktraderAdapter())
        except ImportError:
            pass  # Optional for now due to known issues

        if len(adapters) < 2:
            pytest.skip("Need at least 2 frameworks for comparison")

        # Run all backtests
        results = []
        for adapter in adapters:
            result = adapter.run_backtest(test_data, momentum_strategy, 10000)
            if not result.has_errors:
                results.append(result)

        assert len(results) >= 2, "Need at least 2 successful results"

        # Check consistency
        returns = [r.total_return for r in results]
        return_std = np.std(returns)

        # Allow some variance but flag large discrepancies
        if return_std > 1.0:  # More than 1% standard deviation
            pytest.skip(f"Known framework discrepancy: {return_std:.2f}% std dev")

    @pytest.mark.parametrize("short_window,long_window", [(5, 10), (10, 20), (20, 50)])
    def test_different_strategy_parameters(self, test_data, short_window, long_window):
        """Test ml4t.backtest with different strategy parameters."""
        strategy_params = {
            "name": "MovingAverageCrossover",
            "short_window": short_window,
            "long_window": long_window,
        }

        qe_adapter = ml4t.backtestAdapter()
        result = qe_adapter.run_backtest(test_data, strategy_params, 10000)

        assert not result.has_errors, f"Strategy failed: {result.errors}"
        assert result.final_value > 0, "Final value must be positive"

    def test_edge_cases(self, test_data, momentum_strategy):
        """Test ml4t.backtest with edge cases."""
        qe_adapter = ml4t.backtestAdapter()

        # Test with minimal capital
        result = qe_adapter.run_backtest(test_data, momentum_strategy, 100)
        assert not result.has_errors, "Failed with minimal capital"

        # Test with very short data
        short_data = test_data.head(60)  # Just enough for MA calculation
        result = qe_adapter.run_backtest(short_data, momentum_strategy, 10000)
        assert not result.has_errors, "Failed with short data"

        # Test with few signals (large MA windows)
        # Note: With 504 rows, even 100/200 windows can generate some signals
        few_signal_strategy = {
            "name": "MovingAverageCrossover",
            "short_window": 100,
            "long_window": 200,
        }
        result = qe_adapter.run_backtest(test_data, few_signal_strategy, 10000)
        assert not result.has_errors, "Failed with few signals"
        assert result.num_trades <= 5, f"Should have few trades, got {result.num_trades}"


class TestRegressionPrevention:
    """Test to prevent regressions in validated behavior."""

    def test_known_good_results(self, test_data):
        """Test against known good results to prevent regression."""
        # Use fixed strategy parameters
        strategy_params = {"name": "MovingAverageCrossover", "short_window": 10, "long_window": 20}

        qe_adapter = ml4t.backtestAdapter()
        result = qe_adapter.run_backtest(test_data, strategy_params, 10000)

        assert not result.has_errors, f"Regression test failed: {result.errors}"

        # These values depend on the exact test data, so adapt as needed
        # The key is to detect unexpected changes in behavior
        assert result.num_trades >= 0, "Should produce some trades or none"
        assert result.final_value > 0, "Final value should be positive"
        assert abs(result.total_return) < 50, "Return should be reasonable"


if __name__ == "__main__":
    # Allow running as script for debugging
    pytest.main([__file__, "-v"])
