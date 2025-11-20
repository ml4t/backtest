"""Unit tests for basic risk management rules."""

import pytest
from unittest.mock import Mock

from ml4t.backtest.risk.rules import TimeBasedExit, PriceBasedStopLoss, PriceBasedTakeProfit
from ml4t.backtest.risk.context import RiskContext
from ml4t.backtest.risk.decision import ExitType


class TestTimeBasedExit:
    """Tests for TimeBasedExit rule."""

    def test_init_validates_max_bars(self):
        """Test that max_bars must be >= 1."""
        with pytest.raises(ValueError, match="max_bars must be >= 1"):
            TimeBasedExit(max_bars=0)

        with pytest.raises(ValueError, match="max_bars must be >= 1"):
            TimeBasedExit(max_bars=-5)

        # Valid values should work
        TimeBasedExit(max_bars=1)
        TimeBasedExit(max_bars=60)

    def test_no_position_returns_no_action(self):
        """Test that rule returns NO_ACTION when there's no position."""
        rule = TimeBasedExit(max_bars=60)

        context = Mock(spec=RiskContext)
        context.position_quantity = 0
        context.bars_held = 10
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit
        assert "No position" in decision.reason

    def test_no_time_tracking_returns_no_action(self):
        """Test that rule returns NO_ACTION when bars_held is None."""
        rule = TimeBasedExit(max_bars=60)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.bars_held = None
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit
        assert "No position or no time tracking" in decision.reason

    def test_exit_triggers_when_max_bars_reached(self):
        """Test that exit triggers when bars_held >= max_bars."""
        rule = TimeBasedExit(max_bars=60)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.bars_held = 60
        context.entry_price = 100.0
        context.current_price = 105.0
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert decision.should_exit
        assert decision.exit_type == ExitType.TIME_EXIT
        assert "held 60/60" in decision.reason
        assert decision.metadata["bars_held"] == 60
        assert decision.metadata["max_bars"] == 60

    def test_exit_triggers_when_max_bars_exceeded(self):
        """Test that exit triggers when bars_held > max_bars."""
        rule = TimeBasedExit(max_bars=60)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.bars_held = 61
        context.entry_price = 100.0
        context.current_price = 105.0
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert decision.should_exit
        assert decision.exit_type == ExitType.TIME_EXIT
        assert "held 61/60" in decision.reason

    def test_no_action_when_within_time_limit(self):
        """Test that rule returns NO_ACTION when bars_held < max_bars."""
        rule = TimeBasedExit(max_bars=60)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.bars_held = 59
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit
        assert "within time limit" in decision.reason
        assert "(59/60 bars)" in decision.reason

    def test_priority_is_medium(self):
        """Test that TimeBasedExit has medium priority (5)."""
        rule = TimeBasedExit(max_bars=60)
        assert rule.priority == 5


class TestPriceBasedStopLoss:
    """Tests for PriceBasedStopLoss rule."""

    def test_no_position_returns_no_action(self):
        """Test that rule returns NO_ACTION when there's no position."""
        rule = PriceBasedStopLoss(stop_loss_price=95.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 0
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit
        assert "No position" in decision.reason

    def test_no_stop_price_returns_no_action(self):
        """Test that rule returns NO_ACTION when no stop price configured."""
        rule = PriceBasedStopLoss()  # No fixed stop price

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.stop_loss_price = None  # No position-level stop either
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit
        assert "No stop loss price configured" in decision.reason

    def test_long_position_stop_loss_hit(self):
        """Test stop loss triggers for long position when price <= stop."""
        rule = PriceBasedStopLoss(stop_loss_price=95.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0  # Long
        context.current_price = 94.0  # Below stop
        context.entry_price = 100.0
        context.stop_loss_price = None
        context.mae = -6.0
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert decision.should_exit
        assert decision.exit_type == ExitType.STOP_LOSS
        assert "Stop loss hit" in decision.reason
        assert "price=94.00" in decision.reason
        assert "stop=95.00" in decision.reason
        assert decision.metadata["stop_loss_price"] == 95.0
        assert decision.metadata["current_price"] == 94.0
        assert decision.metadata["position_direction"] == "long"

    def test_long_position_stop_loss_exact(self):
        """Test stop loss triggers for long when price exactly equals stop."""
        rule = PriceBasedStopLoss(stop_loss_price=95.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0  # Long
        context.current_price = 95.0  # Exactly at stop
        context.entry_price = 100.0
        context.stop_loss_price = None
        context.mae = -5.0
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert decision.should_exit
        assert decision.exit_type == ExitType.STOP_LOSS

    def test_long_position_stop_loss_not_hit(self):
        """Test stop loss doesn't trigger for long when price > stop."""
        rule = PriceBasedStopLoss(stop_loss_price=95.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0  # Long
        context.current_price = 96.0  # Above stop
        context.entry_price = 100.0
        context.stop_loss_price = None
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit
        assert "above stop loss" in decision.reason
        assert decision.metadata["distance_to_stop"] == 1.0

    def test_short_position_stop_loss_hit(self):
        """Test stop loss triggers for short position when price >= stop."""
        rule = PriceBasedStopLoss(stop_loss_price=105.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = -100.0  # Short
        context.current_price = 106.0  # Above stop
        context.entry_price = 100.0
        context.stop_loss_price = None
        context.mae = -6.0
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert decision.should_exit
        assert decision.exit_type == ExitType.STOP_LOSS
        assert "Stop loss hit" in decision.reason
        assert decision.metadata["position_direction"] == "short"

    def test_short_position_stop_loss_not_hit(self):
        """Test stop loss doesn't trigger for short when price < stop."""
        rule = PriceBasedStopLoss(stop_loss_price=105.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = -100.0  # Short
        context.current_price = 104.0  # Below stop
        context.entry_price = 100.0
        context.stop_loss_price = None
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit

    def test_uses_position_level_stop_when_fixed_is_none(self):
        """Test rule uses context.stop_loss_price when no fixed price given."""
        rule = PriceBasedStopLoss()  # No fixed stop price

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.current_price = 94.0
        context.entry_price = 100.0
        context.stop_loss_price = 95.0  # From position levels
        context.mae = -6.0
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert decision.should_exit
        assert decision.metadata["stop_loss_price"] == 95.0

    def test_fixed_stop_overrides_position_level(self):
        """Test fixed stop price takes precedence over position level."""
        rule = PriceBasedStopLoss(stop_loss_price=93.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.current_price = 94.0
        context.entry_price = 100.0
        context.stop_loss_price = 95.0  # Position level (ignored)
        context.mae = -6.0
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Should NOT trigger because fixed stop is 93.0 (current 94.0 > 93.0)
        assert not decision.should_exit

    def test_priority_is_high(self):
        """Test that PriceBasedStopLoss has high priority (10)."""
        rule = PriceBasedStopLoss(stop_loss_price=95.0)
        assert rule.priority == 10


class TestPriceBasedTakeProfit:
    """Tests for PriceBasedTakeProfit rule."""

    def test_no_position_returns_no_action(self):
        """Test that rule returns NO_ACTION when there's no position."""
        rule = PriceBasedTakeProfit(take_profit_price=110.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 0
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit
        assert "No position" in decision.reason

    def test_no_take_profit_price_returns_no_action(self):
        """Test that rule returns NO_ACTION when no take profit configured."""
        rule = PriceBasedTakeProfit()  # No fixed TP price

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.take_profit_price = None  # No position-level TP either
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit
        assert "No take profit price configured" in decision.reason

    def test_long_position_take_profit_hit(self):
        """Test take profit triggers for long position when price >= target."""
        rule = PriceBasedTakeProfit(take_profit_price=110.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0  # Long
        context.current_price = 111.0  # Above target
        context.entry_price = 100.0
        context.take_profit_price = None
        context.mfe = 11.0
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert decision.should_exit
        assert decision.exit_type == ExitType.TAKE_PROFIT
        assert "Take profit hit" in decision.reason
        assert "price=111.00" in decision.reason
        assert "target=110.00" in decision.reason
        assert decision.metadata["take_profit_price"] == 110.0
        assert decision.metadata["position_direction"] == "long"

    def test_long_position_take_profit_exact(self):
        """Test take profit triggers for long when price exactly equals target."""
        rule = PriceBasedTakeProfit(take_profit_price=110.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0  # Long
        context.current_price = 110.0  # Exactly at target
        context.entry_price = 100.0
        context.take_profit_price = None
        context.mfe = 10.0
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert decision.should_exit
        assert decision.exit_type == ExitType.TAKE_PROFIT

    def test_long_position_take_profit_not_hit(self):
        """Test take profit doesn't trigger for long when price < target."""
        rule = PriceBasedTakeProfit(take_profit_price=110.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0  # Long
        context.current_price = 109.0  # Below target
        context.entry_price = 100.0
        context.take_profit_price = None
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit
        assert "below take profit" in decision.reason
        assert decision.metadata["distance_to_target"] == 1.0

    def test_short_position_take_profit_hit(self):
        """Test take profit triggers for short position when price <= target."""
        rule = PriceBasedTakeProfit(take_profit_price=95.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = -100.0  # Short
        context.current_price = 94.0  # Below target
        context.entry_price = 100.0
        context.take_profit_price = None
        context.mfe = 6.0
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert decision.should_exit
        assert decision.exit_type == ExitType.TAKE_PROFIT
        assert "Take profit hit" in decision.reason
        assert decision.metadata["position_direction"] == "short"

    def test_short_position_take_profit_not_hit(self):
        """Test take profit doesn't trigger for short when price > target."""
        rule = PriceBasedTakeProfit(take_profit_price=95.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = -100.0  # Short
        context.current_price = 96.0  # Above target
        context.entry_price = 100.0
        context.take_profit_price = None
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit

    def test_uses_position_level_tp_when_fixed_is_none(self):
        """Test rule uses context.take_profit_price when no fixed price given."""
        rule = PriceBasedTakeProfit()  # No fixed TP price

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.current_price = 111.0
        context.entry_price = 100.0
        context.take_profit_price = 110.0  # From position levels
        context.mfe = 11.0
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert decision.should_exit
        assert decision.metadata["take_profit_price"] == 110.0

    def test_fixed_tp_overrides_position_level(self):
        """Test fixed take profit price takes precedence over position level."""
        rule = PriceBasedTakeProfit(take_profit_price=115.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.current_price = 111.0
        context.entry_price = 100.0
        context.take_profit_price = 110.0  # Position level (ignored)
        context.mfe = 11.0
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Should NOT trigger because fixed TP is 115.0 (current 111.0 < 115.0)
        assert not decision.should_exit

    def test_priority_is_medium_high(self):
        """Test that PriceBasedTakeProfit has medium-high priority (8)."""
        rule = PriceBasedTakeProfit(take_profit_price=110.0)
        assert rule.priority == 8
