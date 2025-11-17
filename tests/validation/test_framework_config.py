"""
Test FrameworkConfig and framework adapter configuration support.

This test verifies:
1. FrameworkConfig can be created with all presets
2. All adapters accept FrameworkConfig
3. Configuration is correctly applied to each framework
"""

import pytest
from tests.validation.frameworks.base import FrameworkConfig


class TestFrameworkConfig:
    """Test FrameworkConfig dataclass and presets."""

    def test_default_config(self):
        """Test default FrameworkConfig creation."""
        config = FrameworkConfig()

        assert config.commission_pct == 0.001
        assert config.slippage_pct == 0.0005
        assert config.initial_capital == 100000.0
        assert config.fill_timing == "next_open"
        assert config.backtrader_coo is False
        assert config.backtrader_coc is False
        assert config.vectorbt_accumulate is False
        assert config.fractional_shares is False

    def test_realistic_preset(self):
        """Test realistic() preset."""
        config = FrameworkConfig.realistic()

        assert config.commission_pct == 0.001
        assert config.slippage_pct == 0.0005
        assert config.fill_timing == "next_open"
        assert config.backtrader_coo is False
        assert config.backtrader_coc is False
        assert config.vectorbt_accumulate is False

    def test_backtrader_compatible_preset(self):
        """Test backtrader_compatible() preset."""
        config = FrameworkConfig.backtrader_compatible()

        assert config.fill_timing == "next_open"
        assert config.backtrader_coo is False
        assert config.backtrader_coc is False

    def test_vectorbt_compatible_preset(self):
        """Test vectorbt_compatible() preset (with look-ahead bias warning)."""
        with pytest.warns(UserWarning, match="look-ahead bias"):
            config = FrameworkConfig.vectorbt_compatible()

        assert config.fill_timing == "same_close"
        assert config.vectorbt_accumulate is False
        assert config.fractional_shares is True

    def test_custom_config(self):
        """Test creating custom configuration."""
        config = FrameworkConfig(
            commission_pct=0.002,  # 0.2%
            slippage_pct=0.001,    # 0.1%
            initial_capital=50000,
            fill_timing="next_close",
        )

        assert config.commission_pct == 0.002
        assert config.slippage_pct == 0.001
        assert config.initial_capital == 50000
        assert config.fill_timing == "next_close"

    def test_validation_negative_commission(self):
        """Test validation rejects negative commission."""
        with pytest.raises(ValueError, match="commission_pct must be non-negative"):
            FrameworkConfig(commission_pct=-0.001)

    def test_validation_negative_slippage(self):
        """Test validation rejects negative slippage."""
        with pytest.raises(ValueError, match="slippage_pct must be non-negative"):
            FrameworkConfig(slippage_pct=-0.001)

    def test_validation_zero_capital(self):
        """Test validation rejects zero or negative initial capital."""
        with pytest.raises(ValueError, match="initial_capital must be positive"):
            FrameworkConfig(initial_capital=0)

        with pytest.raises(ValueError, match="initial_capital must be positive"):
            FrameworkConfig(initial_capital=-10000)

    def test_look_ahead_bias_warnings(self):
        """Test warnings for look-ahead bias configurations."""
        # same_close fill timing warning
        with pytest.warns(UserWarning, match="look-ahead bias"):
            FrameworkConfig(fill_timing="same_close")

        # Backtrader COO warning
        with pytest.warns(UserWarning, match="look-ahead bias"):
            FrameworkConfig(backtrader_coo=True)

        # Backtrader COC warning
        with pytest.warns(UserWarning, match="look-ahead bias"):
            FrameworkConfig(backtrader_coc=True)


class TestAdapterConfigSupport:
    """Test that all adapters accept FrameworkConfig."""

    def test_backtrader_adapter_accepts_config(self):
        """Test BacktraderAdapter run_with_signals accepts FrameworkConfig."""
        from tests.validation.frameworks.backtrader_adapter import BacktraderAdapter
        import inspect

        adapter = BacktraderAdapter()
        sig = inspect.signature(adapter.run_with_signals)

        # Check that config parameter exists and has correct type hint
        assert 'config' in sig.parameters
        param = sig.parameters['config']
        assert 'FrameworkConfig' in str(param.annotation)

    def test_vectorbt_adapter_accepts_config(self):
        """Test VectorBTAdapter run_with_signals accepts FrameworkConfig."""
        from tests.validation.frameworks.vectorbt_adapter import VectorBTAdapter
        import inspect

        adapter = VectorBTAdapter()
        sig = inspect.signature(adapter.run_with_signals)

        assert 'config' in sig.parameters
        param = sig.parameters['config']
        assert 'FrameworkConfig' in str(param.annotation)

    def test_qengine_adapter_accepts_config(self):
        """Test BacktestAdapter (ml4t.backtest) run_with_signals accepts FrameworkConfig."""
        from tests.validation.frameworks.qengine_adapter import BacktestAdapter
        import inspect

        adapter = BacktestAdapter()
        sig = inspect.signature(adapter.run_with_signals)

        assert 'config' in sig.parameters
        param = sig.parameters['config']
        assert 'FrameworkConfig' in str(param.annotation)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
