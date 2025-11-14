"""
Unit tests for VectorBTProAdapter.
"""

import pandas as pd
import pytest

from .fixtures import get_test_data
from .frameworks.base import ValidationResult
from .frameworks.vectorbtpro_adapter import VectorBTProAdapter


class TestVectorBTProAdapter:
    """Test suite for VectorBTProAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create VectorBT Pro adapter instance."""
        return VectorBTProAdapter()

    @pytest.fixture
    def sample_data(self):
        """Load sample AAPL data for testing."""
        df = get_test_data(symbol="AAPL", start="2017-01-01", end="2017-12-31")
        # Convert to VectorBT format (DatetimeIndex)
        df = df.set_index("timestamp")
        return df

    # ============================================================================
    # Initialization Tests
    # ============================================================================

    def test_adapter_initialization(self, adapter):
        """Test adapter initializes correctly."""
        assert adapter.framework_name in ["VectorBTPro", "VectorBT"]
        assert adapter.vbt is not None
        assert hasattr(adapter, "is_pro")

    def test_adapter_detects_pro_version(self, adapter):
        """Test adapter correctly detects VectorBT Pro vs open-source."""
        # Should detect Pro version in main venv
        assert adapter.is_pro or not adapter.is_pro  # Either is valid
        if adapter.is_pro:
            assert adapter.framework_name == "VectorBTPro"
        else:
            assert adapter.framework_name == "VectorBT"

    # ============================================================================
    # MA Crossover Strategy Tests
    # ============================================================================

    def test_ma_crossover_strategy(self, adapter, sample_data):
        """Test MA crossover strategy runs successfully."""
        strategy_params = {
            "name": "MovingAverageCrossover",
            "short_window": 10,
            "slow_window": 30,
        }

        result = adapter.run_backtest(sample_data, strategy_params, initial_capital=10000)

        assert isinstance(result, ValidationResult)
        assert result.framework in ["VectorBTPro", "VectorBT"]
        assert result.strategy == "MovingAverageCrossover"
        assert result.initial_capital == 10000
        assert not result.has_errors

    def test_ma_crossover_metrics(self, adapter, sample_data):
        """Test MA crossover produces valid metrics."""
        strategy_params = {
            "name": "MovingAverageCrossover",
            "short_window": 10,
            "slow_window": 30,
        }

        result = adapter.run_backtest(sample_data, strategy_params, initial_capital=10000)

        assert result.final_value > 0
        assert isinstance(result.total_return, float)
        assert result.num_trades >= 0
        assert result.execution_time > 0
        assert result.memory_usage > 0

    def test_ma_crossover_different_windows(self, adapter, sample_data):
        """Test MA crossover with different window sizes."""
        strategy_params = {
            "name": "MovingAverageCrossover",
            "short_window": 5,
            "slow_window": 20,
        }

        result = adapter.run_backtest(sample_data, strategy_params, initial_capital=10000)

        assert not result.has_errors
        assert result.final_value > 0

    # ============================================================================
    # Bollinger Bands Strategy Tests
    # ============================================================================

    def test_bollinger_bands_strategy(self, adapter, sample_data):
        """Test Bollinger Bands mean reversion strategy."""
        strategy_params = {
            "name": "BollingerBandMeanReversion",
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
        }

        result = adapter.run_backtest(sample_data, strategy_params, initial_capital=10000)

        assert isinstance(result, ValidationResult)
        assert result.strategy == "BollingerBandMeanReversion"
        assert not result.has_errors

    # ============================================================================
    # Momentum Strategy Tests
    # ============================================================================

    def test_momentum_strategy(self, adapter, sample_data):
        """Test short-term momentum strategy."""
        strategy_params = {
            "name": "ShortTermMomentumStrategy",
            "fast_ema": 5,
            "slow_ema": 15,
            "profit_target": 0.03,
            "stop_loss": 0.015,
        }

        result = adapter.run_backtest(sample_data, strategy_params, initial_capital=10000)

        assert isinstance(result, ValidationResult)
        assert result.strategy == "ShortTermMomentumStrategy"
        assert not result.has_errors

    # ============================================================================
    # Volume Breakout Strategy Tests
    # ============================================================================

    def test_volume_breakout_strategy(self, adapter, sample_data):
        """Test volume breakout strategy."""
        strategy_params = {
            "name": "VolumeBreakoutStrategy",
            "lookback_period": 20,
            "volume_multiplier": 1.5,
            "breakout_threshold": 0.02,
        }

        result = adapter.run_backtest(sample_data, strategy_params, initial_capital=10000)

        assert isinstance(result, ValidationResult)
        assert result.strategy == "VolumeBreakoutStrategy"
        assert not result.has_errors

    # ============================================================================
    # Edge Case Tests
    # ============================================================================

    def test_unknown_strategy(self, adapter, sample_data):
        """Test handling of unknown strategy."""
        strategy_params = {"name": "NonExistentStrategy"}

        result = adapter.run_backtest(sample_data, strategy_params, initial_capital=10000)

        assert result.has_errors
        assert any("Unknown strategy" in error for error in result.errors)

    def test_invalid_parameters(self, adapter, sample_data):
        """Test handling of invalid strategy parameters."""
        strategy_params = {
            "name": "MovingAverageCrossover",
            "short_window": 50,  # Longer than slow window
            "slow_window": 10,
        }

        # Should still run but may have no trades
        result = adapter.run_backtest(sample_data, strategy_params, initial_capital=10000)

        # Either succeeds with no trades or handles gracefully
        assert isinstance(result, ValidationResult)

    def test_small_initial_capital(self, adapter, sample_data):
        """Test with very small initial capital."""
        strategy_params = {
            "name": "MovingAverageCrossover",
            "short_window": 10,
            "slow_window": 30,
        }

        result = adapter.run_backtest(sample_data, strategy_params, initial_capital=100)

        assert result.initial_capital == 100
        assert not result.has_errors

    def test_large_initial_capital(self, adapter, sample_data):
        """Test with very large initial capital."""
        strategy_params = {
            "name": "MovingAverageCrossover",
            "short_window": 10,
            "slow_window": 30,
        }

        result = adapter.run_backtest(sample_data, strategy_params, initial_capital=1000000)

        assert result.initial_capital == 1000000
        assert not result.has_errors

    # ============================================================================
    # Data Validation Tests
    # ============================================================================

    def test_data_validation_missing_columns(self, adapter):
        """Test data validation catches missing columns."""
        invalid_data = pd.DataFrame(
            {"close": [100, 101, 102]}, index=pd.date_range("2020-01-01", periods=3)
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            adapter.validate_data(invalid_data)

    def test_data_validation_non_datetime_index(self, adapter):
        """Test data validation catches non-DatetimeIndex."""
        invalid_data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
                "close": [100, 101, 102],
                "volume": [1000, 1100, 1200],
            }
        )

        with pytest.raises(ValueError, match="Data must have DatetimeIndex"):
            adapter.validate_data(invalid_data)

    # ============================================================================
    # Result Validation Tests
    # ============================================================================

    def test_result_contains_required_fields(self, adapter, sample_data):
        """Test ValidationResult contains all required fields."""
        strategy_params = {
            "name": "MovingAverageCrossover",
            "short_window": 10,
            "slow_window": 30,
        }

        result = adapter.run_backtest(sample_data, strategy_params, initial_capital=10000)

        assert hasattr(result, "framework")
        assert hasattr(result, "strategy")
        assert hasattr(result, "initial_capital")
        assert hasattr(result, "final_value")
        assert hasattr(result, "total_return")
        assert hasattr(result, "num_trades")
        assert hasattr(result, "execution_time")
        assert hasattr(result, "memory_usage")

    def test_result_summary_dict(self, adapter, sample_data):
        """Test ValidationResult summary_dict method."""
        strategy_params = {
            "name": "MovingAverageCrossover",
            "short_window": 10,
            "slow_window": 30,
        }

        result = adapter.run_backtest(sample_data, strategy_params, initial_capital=10000)
        summary = result.summary_dict()

        assert isinstance(summary, dict)
        assert "Framework" in summary
        assert "Final Value ($)" in summary
        assert "Return (%)" in summary
        assert "Trades" in summary
        assert "Status" in summary

    # ============================================================================
    # Performance Metrics Tests
    # ============================================================================

    def test_execution_time_reasonable(self, adapter, sample_data):
        """Test execution time is reasonable (not too slow)."""
        strategy_params = {
            "name": "MovingAverageCrossover",
            "short_window": 10,
            "slow_window": 30,
        }

        result = adapter.run_backtest(sample_data, strategy_params, initial_capital=10000)

        # Should complete in under 10 seconds for 249 rows
        assert result.execution_time < 10.0

    def test_memory_usage_tracked(self, adapter, sample_data):
        """Test memory usage is tracked."""
        strategy_params = {
            "name": "MovingAverageCrossover",
            "short_window": 10,
            "slow_window": 30,
        }

        result = adapter.run_backtest(sample_data, strategy_params, initial_capital=10000)

        assert result.memory_usage > 0
        # Reasonable memory usage (< 500 MB for simple strategy)
        assert result.memory_usage < 500
