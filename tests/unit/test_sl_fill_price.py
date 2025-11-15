"""Test that SL orders fill at stop_price (with slippage), not at low.

This is the CRITICAL behavior from TASK-006: SL exits at SL level, NOT at low.
This test validates the fill_simulator integration.
"""

import pytest
from datetime import datetime

from ml4t.backtest.core.types import OrderSide, OrderType
from ml4t.backtest.data.asset_registry import AssetRegistry, AssetSpec
from ml4t.backtest.execution.fill_simulator import FillSimulator
from ml4t.backtest.execution.order import Order
from ml4t.backtest.execution.slippage import PercentageSlippage


class TestSLFillPrice:
    """Test SL fill price calculation matches VectorBT (exit at stop level, not low)."""

    @pytest.fixture
    def asset_registry(self):
        """Create basic asset registry."""
        registry = AssetRegistry()
        registry.register(
            spec=AssetSpec(
                asset_id="BTC",
                asset_type="crypto",
                tick_size=0.01,
                lot_size=1,
                multiplier=1.0,
                margin_requirement=1.0,
            )
        )
        return registry

    @pytest.fixture
    def fill_simulator(self, asset_registry):
        """Create fill simulator with percentage slippage."""
        slippage_model = PercentageSlippage(slippage_pct=0.0002, min_slippage=0.0)  # 0.02%
        return FillSimulator(
            asset_registry=asset_registry,
            slippage_model=slippage_model,
        )

    def test_sl_fills_at_stop_price_not_low(self, fill_simulator):
        """CRITICAL: SL should fill at stop_price (with slippage), NOT at low.

        From TASK-006:
        SL level: $44,679.38
        Low:      $44,605.00 (worse by $74.38)
        Exit:     $44,679.38 * (1 - 0.0002) = $44,670.44  ← At SL level, not low
        """
        sl_level = 44679.38
        low = 44605.0  # Worse price (bar went below SL)
        close = 44690.0  # Close recovered above SL
        high = 44730.0

        sl_order = Order(
            asset_id="BTC",
            order_type=OrderType.STOP,
            side=OrderSide.SELL,  # Exit long
            quantity=0.002181,
            stop_price=sl_level,
        )

        # Try to fill with OHLC data
        result = fill_simulator.try_fill_order(
            order=sl_order,
            current_cash=1000.0,
            current_position=0.002181,  # Have position to sell
            timestamp=datetime(2024, 1, 3, 11, 50),
            high=high,
            low=low,
            close=close,
        )

        assert result is not None, "SL should trigger when low <= stop_price"

        # CRITICAL: Fill should be at SL level (with slippage), NOT at low
        expected_fill = sl_level * (1 - 0.0002)  # $44,670.44
        wrong_fill = low * (1 - 0.0002)  # $44,596.09

        assert result.fill_price == pytest.approx(expected_fill, abs=0.1), \
            f"SL must fill at stop_price={sl_level} with slippage, not at low={low}"

        # Verify it's NOT filling at low
        assert abs(result.fill_price - wrong_fill) > 50, \
            f"SL should NOT fill at low={low}, should fill at stop={sl_level}"

    def test_sl_trigger_but_not_fill_if_no_low(self, fill_simulator):
        """SL should NOT trigger if low > stop_price."""
        sl_level = 44679.38
        low = 44680.0  # Above SL - should NOT trigger
        close = 44690.0
        high = 44730.0

        sl_order = Order(
            asset_id="BTC",
            order_type=OrderType.STOP,
            side=OrderSide.SELL,
            quantity=0.002181,
            stop_price=sl_level,
        )

        result = fill_simulator.try_fill_order(
            order=sl_order,
            current_cash=1000.0,
            current_position=0.002181,
            timestamp=datetime(2024, 1, 3, 11, 50),
            high=high,
            low=low,
            close=close,
        )

        assert result is None, f"SL should NOT trigger when low={low} > stop_price={sl_level}"

    def test_sl_slippage_direction(self, fill_simulator):
        """Verify slippage is applied in the UNFAVORABLE direction for SL."""
        sl_level = 44679.38
        low = 44600.0
        close = 44690.0
        high = 44730.0

        sl_order = Order(
            asset_id="BTC",
            order_type=OrderType.STOP,
            side=OrderSide.SELL,  # Selling - want higher price
            quantity=0.002181,
            stop_price=sl_level,
        )

        result = fill_simulator.try_fill_order(
            order=sl_order,
            current_cash=1000.0,
            current_position=0.002181,
            timestamp=datetime(2024, 1, 3, 11, 50),
            high=high,
            low=low,
            close=close,
        )

        assert result is not None

        # For SELL, slippage should reduce price (unfavorable)
        # Expected: stop_price * (1 - slippage_rate)
        expected_fill = sl_level * (1 - 0.0002)

        assert result.fill_price < sl_level, \
            "SL sell should have unfavorable slippage (receive less)"

        assert result.fill_price == pytest.approx(expected_fill, abs=0.1)

    def test_sl_short_cover_fills_at_stop_not_high(self, fill_simulator):
        """For short positions, SL (buy to cover) should fill at stop, not high."""
        sl_level = 46971.56  # SL for short (higher than entry)
        high = 47100.0  # Worse price for covering short
        low = 46800.0
        close = 46900.0

        sl_order = Order(
            asset_id="BTC",
            order_type=OrderType.STOP,
            side=OrderSide.BUY,  # Cover short
            quantity=0.002181,
            stop_price=sl_level,
        )

        result = fill_simulator.try_fill_order(
            order=sl_order,
            current_cash=10000.0,  # Enough cash
            current_position=-0.002181,  # Short position
            timestamp=datetime(2024, 1, 3, 11, 50),
            high=high,
            low=low,
            close=close,
        )

        assert result is not None, "Short SL should trigger when high >= stop_price"

        # Should fill at stop_price (with slippage), NOT at high
        expected_fill = sl_level * (1 + 0.0002)  # Buy - pay more
        wrong_fill = high * (1 + 0.0002)

        assert result.fill_price == pytest.approx(expected_fill, abs=0.1), \
            f"Short SL must fill at stop_price={sl_level}, not at high={high}"

    def test_sl_realistic_scenario_from_task006(self, fill_simulator):
        """Test exact scenario from TASK-006 empirical test.

        Entry price:        $45,834.17
        Base price:         $45,825.00
        SL level:           $44,679.38  (= $45,825 * 0.975)
        SL exit price:      $44,670.44  (= $44,679.38 * 0.9998)
        Low at exit bar:    $44,605.00 (SL triggered, but exit at level)
        """
        sl_level = 44679.38
        exit_slippage_rate = 0.0002

        # Bar that triggers SL
        high = 44730.0
        low = 44605.0  # Below SL, triggers it
        close = 44690.0

        sl_order = Order(
            asset_id="BTC",
            order_type=OrderType.STOP,
            side=OrderSide.SELL,
            quantity=0.002181,
            stop_price=sl_level,
        )

        result = fill_simulator.try_fill_order(
            order=sl_order,
            current_cash=1000.0,
            current_position=0.002181,
            timestamp=datetime(2024, 1, 3, 11, 50),
            high=high,
            low=low,
            close=close,
        )

        assert result is not None

        # Exact expected fill from TASK-006
        expected_fill = 44670.44  # sl_level * (1 - 0.0002)

        assert result.fill_price == pytest.approx(expected_fill, abs=0.1), \
            f"Expected exact fill from TASK-006: {expected_fill}"

        # Verify savings compared to filling at low
        low_fill = low * (1 - exit_slippage_rate)
        savings = result.fill_price - low_fill

        assert savings > 60, \
            f"Filling at SL level saves ${savings:.2f} vs filling at low (expected ~$65)"
