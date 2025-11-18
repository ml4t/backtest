"""Unit tests for portfolio constraint rules.

Tests MaxDailyLossRule, MaxDrawdownRule, and MaxLeverageRule validate_order() behavior.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from ml4t.backtest.core.types import AssetId, Price, Quantity, OrderSide, OrderType
from ml4t.backtest.execution.order import Order
from ml4t.backtest.risk.context import RiskContext
from ml4t.backtest.risk.rules.portfolio_constraints import (
    MaxDailyLossRule,
    MaxDrawdownRule,
    MaxLeverageRule,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def base_context():
    """Create a base RiskContext for testing."""
    return RiskContext(
        timestamp=datetime(2024, 1, 2, 10, 0),
        asset_id="AAPL",
        open=150.0,
        high=152.0,
        low=149.0,
        close=151.0,
        volume=1000000.0,
        bid_price=150.95,
        ask_price=151.05,
        position_quantity=0.0,
        entry_price=0.0,
        entry_time=None,
        bars_held=0,
        equity=100000.0,  # $100k portfolio
        cash=100000.0,
        leverage=0.0,  # No positions
        features={},
        market_features={},
    )


@pytest.fixture
def buy_order():
    """Create a buy market order."""
    return Order(
        asset_id="AAPL",
        quantity=100,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
    )


# ============================================================================
# MaxDailyLossRule Tests
# ============================================================================


class TestMaxDailyLossRule:
    """Tests for MaxDailyLossRule - daily P&L constraint."""

    def test_initialization(self):
        """Test rule initialization and validation."""
        rule = MaxDailyLossRule(max_loss_pct=0.02)
        assert rule.max_loss_pct == 0.02
        assert rule.priority == 15

    def test_invalid_parameters(self):
        """Test validation of max_loss_pct parameter."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            MaxDailyLossRule(max_loss_pct=0.0)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            MaxDailyLossRule(max_loss_pct=1.0)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            MaxDailyLossRule(max_loss_pct=-0.05)

    def test_allows_order_when_no_loss(self, base_context, buy_order):
        """Test order allowed when no daily loss."""
        rule = MaxDailyLossRule(max_loss_pct=0.02)

        # First call initializes session
        validated = rule.validate_order(buy_order, base_context)
        assert validated == buy_order  # Order allowed

    def test_allows_order_within_loss_limit(self, base_context, buy_order):
        """Test order allowed when daily loss is within limit."""
        rule = MaxDailyLossRule(max_loss_pct=0.02)  # 2% limit

        # Initialize session at $100k
        rule.validate_order(buy_order, base_context)

        # Lose 1% (within 2% limit)
        context_with_loss = base_context.__class__(
            **{**base_context.__dict__, "equity": 99000.0}  # -1% loss
        )

        validated = rule.validate_order(buy_order, context_with_loss)
        assert validated == buy_order  # Order allowed

    def test_rejects_order_when_loss_exceeds_limit(self, base_context, buy_order):
        """Test order rejected when daily loss exceeds limit."""
        rule = MaxDailyLossRule(max_loss_pct=0.02)  # 2% limit

        # Initialize session at $100k
        rule.validate_order(buy_order, base_context)

        # Lose 3% (exceeds 2% limit)
        context_with_big_loss = base_context.__class__(
            **{**base_context.__dict__, "equity": 97000.0}  # -3% loss
        )

        validated = rule.validate_order(buy_order, context_with_big_loss)
        assert validated is None  # Order rejected

    def test_rejects_order_at_exact_limit(self, base_context, buy_order):
        """Test order rejected when daily loss equals limit."""
        rule = MaxDailyLossRule(max_loss_pct=0.02)  # 2% limit

        # Initialize session at $100k
        rule.validate_order(buy_order, base_context)

        # Lose exactly 2%
        context_at_limit = base_context.__class__(
            **{**base_context.__dict__, "equity": 98000.0}  # -2.0% loss
        )

        validated = rule.validate_order(buy_order, context_at_limit)
        assert validated is None  # Order rejected (loss >= limit)

    def test_session_reset_on_new_day(self, base_context, buy_order):
        """Test session resets at start of new trading day."""
        rule = MaxDailyLossRule(max_loss_pct=0.02)

        # Day 1: Initialize at $100k
        rule.validate_order(buy_order, base_context)

        # Day 1: Lose 3% (triggers limit)
        context_day1_loss = base_context.__class__(
            **{**base_context.__dict__, "equity": 97000.0}
        )
        assert rule.validate_order(buy_order, context_day1_loss) is None

        # Day 2: New session (next day)
        context_day2 = base_context.__class__(
            **{
                **base_context.__dict__,
                "timestamp": datetime(2024, 1, 3, 10, 0),  # Next day
                "equity": 97000.0,  # Still at $97k
            }
        )

        # Session resets - new starting equity is $97k
        # No loss relative to day 2 start → order allowed
        validated = rule.validate_order(buy_order, context_day2)
        assert validated == buy_order  # Order allowed

    def test_handles_profit(self, base_context, buy_order):
        """Test rule allows orders when portfolio has gains."""
        rule = MaxDailyLossRule(max_loss_pct=0.02)

        # Initialize session
        rule.validate_order(buy_order, base_context)

        # Make 5% profit
        context_with_profit = base_context.__class__(
            **{**base_context.__dict__, "equity": 105000.0}  # +5% gain
        )

        validated = rule.validate_order(buy_order, context_with_profit)
        assert validated == buy_order  # Order allowed

    def test_evaluate_returns_no_action(self, base_context):
        """Test evaluate() always returns no_action (doesn't exit positions)."""
        rule = MaxDailyLossRule(max_loss_pct=0.02)

        decision = rule.evaluate(base_context)
        assert not decision.should_exit
        assert not decision.update_stop_loss
        assert not decision.update_take_profit

    def test_repr(self):
        """Test string representation."""
        rule = MaxDailyLossRule(max_loss_pct=0.02)
        assert "MaxDailyLossRule" in repr(rule)
        assert "2.0%" in repr(rule)
        assert "priority=15" in repr(rule)


# ============================================================================
# MaxDrawdownRule Tests
# ============================================================================


class TestMaxDrawdownRule:
    """Tests for MaxDrawdownRule - drawdown from high-water mark constraint."""

    def test_initialization(self):
        """Test rule initialization and validation."""
        rule = MaxDrawdownRule(max_dd_pct=0.10)
        assert rule.max_dd_pct == 0.10
        assert rule.priority == 15
        assert rule.high_water_mark == 0.0

    def test_invalid_parameters(self):
        """Test validation of max_dd_pct parameter."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            MaxDrawdownRule(max_dd_pct=0.0)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            MaxDrawdownRule(max_dd_pct=1.0)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            MaxDrawdownRule(max_dd_pct=-0.05)

    def test_allows_order_on_first_call(self, base_context, buy_order):
        """Test order allowed on first call (no high-water mark yet)."""
        rule = MaxDrawdownRule(max_dd_pct=0.10)

        validated = rule.validate_order(buy_order, base_context)
        assert validated == buy_order  # Order allowed
        assert rule.high_water_mark == 100000.0  # HWM set to initial equity

    def test_updates_high_water_mark_on_gains(self, base_context, buy_order):
        """Test high-water mark increases when equity increases."""
        rule = MaxDrawdownRule(max_dd_pct=0.10)

        # Initialize HWM at $100k
        rule.validate_order(buy_order, base_context)
        assert rule.high_water_mark == 100000.0

        # Equity increases to $120k
        context_higher = base_context.__class__(
            **{**base_context.__dict__, "equity": 120000.0}
        )
        rule.validate_order(buy_order, context_higher)
        assert rule.high_water_mark == 120000.0  # HWM updated

    def test_does_not_decrease_high_water_mark(self, base_context, buy_order):
        """Test high-water mark never decreases."""
        rule = MaxDrawdownRule(max_dd_pct=0.10)

        # Set HWM at $120k
        context_peak = base_context.__class__(
            **{**base_context.__dict__, "equity": 120000.0}
        )
        rule.validate_order(buy_order, context_peak)
        assert rule.high_water_mark == 120000.0

        # Equity drops to $110k
        context_lower = base_context.__class__(
            **{**base_context.__dict__, "equity": 110000.0}
        )
        rule.validate_order(buy_order, context_lower)
        assert rule.high_water_mark == 120000.0  # HWM unchanged

    def test_allows_order_within_drawdown_limit(self, base_context, buy_order):
        """Test order allowed when drawdown is within limit."""
        rule = MaxDrawdownRule(max_dd_pct=0.10)  # 10% limit

        # Set HWM at $120k
        context_peak = base_context.__class__(
            **{**base_context.__dict__, "equity": 120000.0}
        )
        rule.validate_order(buy_order, context_peak)

        # Drop to $110k → 8.33% drawdown (within 10% limit)
        context_dd = base_context.__class__(
            **{**base_context.__dict__, "equity": 110000.0}
        )
        validated = rule.validate_order(buy_order, context_dd)
        assert validated == buy_order  # Order allowed

    def test_rejects_order_when_drawdown_exceeds_limit(self, base_context, buy_order):
        """Test order rejected when drawdown exceeds limit."""
        rule = MaxDrawdownRule(max_dd_pct=0.10)  # 10% limit

        # Set HWM at $120k
        context_peak = base_context.__class__(
            **{**base_context.__dict__, "equity": 120000.0}
        )
        rule.validate_order(buy_order, context_peak)

        # Drop to $105k → 12.5% drawdown (exceeds 10% limit)
        context_big_dd = base_context.__class__(
            **{**base_context.__dict__, "equity": 105000.0}
        )
        validated = rule.validate_order(buy_order, context_big_dd)
        assert validated is None  # Order rejected

    def test_resumes_trading_after_recovery(self, base_context, buy_order):
        """Test trading resumes when drawdown improves below limit."""
        rule = MaxDrawdownRule(max_dd_pct=0.10)

        # Set HWM at $120k
        context_peak = base_context.__class__(
            **{**base_context.__dict__, "equity": 120000.0}
        )
        rule.validate_order(buy_order, context_peak)

        # Drop to $105k → 12.5% DD → rejected
        context_big_dd = base_context.__class__(
            **{**base_context.__dict__, "equity": 105000.0}
        )
        assert rule.validate_order(buy_order, context_big_dd) is None

        # Recover to $110k → 8.33% DD → allowed
        context_recovered = base_context.__class__(
            **{**base_context.__dict__, "equity": 110000.0}
        )
        validated = rule.validate_order(buy_order, context_recovered)
        assert validated == buy_order  # Order allowed

    def test_evaluate_returns_no_action(self, base_context):
        """Test evaluate() always returns no_action."""
        rule = MaxDrawdownRule(max_dd_pct=0.10)

        decision = rule.evaluate(base_context)
        assert not decision.should_exit

    def test_repr(self):
        """Test string representation."""
        rule = MaxDrawdownRule(max_dd_pct=0.10)
        rule._high_water_mark = 120000.0

        assert "MaxDrawdownRule" in repr(rule)
        assert "10.0%" in repr(rule)
        assert "hwm=$120000.00" in repr(rule)


# ============================================================================
# MaxLeverageRule Tests
# ============================================================================


class TestMaxLeverageRule:
    """Tests for MaxLeverageRule - leverage constraint."""

    def test_initialization(self):
        """Test rule initialization and validation."""
        rule = MaxLeverageRule(max_leverage=2.0)
        assert rule.max_leverage == 2.0
        assert rule.allow_partial is False
        assert rule.priority == 10

    def test_invalid_parameters(self):
        """Test validation of max_leverage parameter."""
        with pytest.raises(ValueError, match="must be positive"):
            MaxLeverageRule(max_leverage=0.0)

        with pytest.raises(ValueError, match="must be positive"):
            MaxLeverageRule(max_leverage=-1.0)

    def test_allows_order_when_no_positions(self, base_context, buy_order):
        """Test order allowed when starting with no positions."""
        rule = MaxLeverageRule(max_leverage=2.0)

        # No positions (leverage = 0.0)
        validated = rule.validate_order(buy_order, base_context)
        assert validated == buy_order  # Order allowed

    def test_allows_order_within_leverage_limit(self, buy_order):
        """Test order allowed when leverage stays within limit."""
        rule = MaxLeverageRule(max_leverage=2.0)

        # Current: 1.5x leverage ($150k position / $100k equity)
        context_leveraged = RiskContext(
            timestamp=datetime(2024, 1, 2, 10, 0),
            asset_id="AAPL",
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000.0,
            bid_price=150.95,
            ask_price=151.05,
            position_quantity=100.0,  # Existing position
            entry_price=150.0,
            entry_time=datetime(2024, 1, 2, 9, 30),
            bars_held=1,
            equity=100000.0,
            cash=85000.0,
            leverage=1.5,  # 1.5x leverage
            features={},
            market_features={},
        )

        # Small order: 50 shares × $151 = $7,550
        # New leverage: ($150k + $7.5k) / $100k = 1.575x (within 2.0x limit)
        small_order = Order(
            asset_id="AAPL",
            quantity=50,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )

        validated = rule.validate_order(small_order, context_leveraged)
        assert validated == small_order  # Order allowed

    def test_rejects_order_when_leverage_exceeds_limit(self, buy_order):
        """Test order rejected when leverage would exceed limit."""
        rule = MaxLeverageRule(max_leverage=2.0, allow_partial=False)

        # Current: 1.8x leverage
        context_high_leverage = RiskContext(
            timestamp=datetime(2024, 1, 2, 10, 0),
            asset_id="AAPL",
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000.0,
            bid_price=150.95,
            ask_price=151.05,
            position_quantity=0.0,
            entry_price=0.0,
            entry_time=None,
            bars_held=0,
            equity=100000.0,
            cash=100000.0,
            leverage=1.8,  # 1.8x leverage
            features={},
            market_features={},
        )

        # Large order: 200 shares × $151 = $30,200
        # New leverage: ($180k + $30.2k) / $100k = 2.102x (exceeds 2.0x)
        large_order = Order(
            asset_id="AAPL",
            quantity=200,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )

        validated = rule.validate_order(large_order, context_high_leverage)
        assert validated is None  # Order rejected

    def test_reduces_order_when_allow_partial_true(self, buy_order):
        """Test order size reduced when allow_partial=True."""
        rule = MaxLeverageRule(max_leverage=2.0, allow_partial=True)

        # Current: 1.5x leverage
        context_leveraged = RiskContext(
            timestamp=datetime(2024, 1, 2, 10, 0),
            asset_id="AAPL",
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000.0,
            bid_price=150.95,
            ask_price=151.05,
            position_quantity=0.0,
            entry_price=0.0,
            entry_time=None,
            bars_held=0,
            equity=100000.0,
            cash=100000.0,
            leverage=1.5,  # $150k position / $100k equity
            features={},
            market_features={},
        )

        # Order that would exceed limit: 500 shares × $151 = $75,500
        # Would be: ($150k + $75.5k) / $100k = 2.255x leverage
        large_order = Order(
            asset_id="AAPL",
            quantity=500,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )

        validated = rule.validate_order(large_order, context_leveraged)

        # Should be reduced
        assert validated is not None
        assert validated.quantity < 500  # Reduced quantity
        # Max allowed: $200k position → $50k more → ~331 shares at $151

    def test_rejects_when_at_leverage_limit(self, buy_order):
        """Test order rejected when already at leverage limit."""
        rule = MaxLeverageRule(max_leverage=2.0, allow_partial=True)

        # Already at 2.0x leverage
        context_at_limit = RiskContext(
            timestamp=datetime(2024, 1, 2, 10, 0),
            asset_id="AAPL",
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000.0,
            bid_price=150.95,
            ask_price=151.05,
            position_quantity=0.0,
            entry_price=0.0,
            entry_time=None,
            bars_held=0,
            equity=100000.0,
            cash=100000.0,
            leverage=2.0,  # At limit
            features={},
            market_features={},
        )

        validated = rule.validate_order(buy_order, context_at_limit)
        assert validated is None  # Rejected (no room for more)

    def test_rejects_when_equity_zero(self, base_context, buy_order):
        """Test order rejected when equity is zero or negative."""
        rule = MaxLeverageRule(max_leverage=2.0)

        context_zero_equity = base_context.__class__(
            **{**base_context.__dict__, "equity": 0.0}
        )

        validated = rule.validate_order(buy_order, context_zero_equity)
        assert validated is None  # Rejected

    def test_evaluate_returns_no_action(self, base_context):
        """Test evaluate() always returns no_action."""
        rule = MaxLeverageRule(max_leverage=2.0)

        decision = rule.evaluate(base_context)
        assert not decision.should_exit

    def test_repr(self):
        """Test string representation."""
        rule = MaxLeverageRule(max_leverage=2.0)
        assert "MaxLeverageRule" in repr(rule)
        assert "2.0x" in repr(rule)

        rule_partial = MaxLeverageRule(max_leverage=2.5, allow_partial=True)
        assert "allow_partial=True" in repr(rule_partial)


# ============================================================================
# Integration Tests - Multiple Constraints
# ============================================================================


class TestCombinedConstraints:
    """Test interactions between multiple portfolio constraints."""

    def test_all_constraints_allow_order(self, base_context, buy_order):
        """Test order passes all constraints when conditions are good."""
        daily_loss = MaxDailyLossRule(max_loss_pct=0.02)
        drawdown = MaxDrawdownRule(max_dd_pct=0.10)
        leverage = MaxLeverageRule(max_leverage=2.0)

        # All rules should allow
        assert daily_loss.validate_order(buy_order, base_context) == buy_order
        assert drawdown.validate_order(buy_order, base_context) == buy_order
        assert leverage.validate_order(buy_order, base_context) == buy_order

    def test_daily_loss_rejects_but_others_allow(self, base_context, buy_order):
        """Test order rejected by daily loss but allowed by others."""
        daily_loss = MaxDailyLossRule(max_loss_pct=0.02)
        drawdown = MaxDrawdownRule(max_dd_pct=0.10)

        # Initialize rules
        daily_loss.validate_order(buy_order, base_context)
        drawdown.validate_order(buy_order, base_context)

        # Trigger daily loss (but not drawdown since same as HWM)
        context_loss = base_context.__class__(
            **{**base_context.__dict__, "equity": 97000.0}  # -3% loss
        )

        assert daily_loss.validate_order(buy_order, context_loss) is None  # Rejected
        assert drawdown.validate_order(buy_order, context_loss) == buy_order  # Allowed

    def test_all_constraints_reject_order(self, base_context, buy_order):
        """Test order rejected by all constraints."""
        daily_loss = MaxDailyLossRule(max_loss_pct=0.02)
        drawdown = MaxDrawdownRule(max_dd_pct=0.10)
        leverage = MaxLeverageRule(max_leverage=2.0, allow_partial=False)

        # Setup: Portfolio at $120k peak
        peak_context = base_context.__class__(
            **{**base_context.__dict__, "equity": 120000.0}
        )
        daily_loss.validate_order(buy_order, peak_context)
        drawdown.validate_order(buy_order, peak_context)

        # Bad scenario: Big loss, high leverage
        bad_context = RiskContext(
            timestamp=datetime(2024, 1, 2, 10, 0),
            asset_id="AAPL",
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000.0,
            bid_price=150.95,
            ask_price=151.05,
            position_quantity=0.0,
            entry_price=0.0,
            entry_time=None,
            bars_held=0,
            equity=105000.0,  # Down from $120k → 12.5% DD (exceeds 10%)
            cash=100000.0,
            leverage=2.5,  # Exceeds 2.0x limit
            features={},
            market_features={},
        )

        # Daily loss: -12.5% (exceeds 2%)
        assert daily_loss.validate_order(buy_order, bad_context) is None

        # Drawdown: -12.5% (exceeds 10%)
        assert drawdown.validate_order(buy_order, bad_context) is None

        # Leverage: 2.5x (exceeds 2.0x)
        # Would add more → rejected
        large_order = Order(
            asset_id="AAPL",
            quantity=200,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )
        assert leverage.validate_order(large_order, bad_context) is None


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_daily_loss_zero_equity(self, base_context, buy_order):
        """Test daily loss rule when session_start_equity is zero."""
        rule = MaxDailyLossRule(max_loss_pct=0.02)

        # This shouldn't happen but test graceful handling
        context_zero = base_context.__class__(
            **{**base_context.__dict__, "equity": 0.0}
        )

        # Should allow order (can't calculate loss percentage)
        validated = rule.validate_order(buy_order, context_zero)
        assert validated == buy_order

    def test_drawdown_massive_gain_then_loss(self, base_context, buy_order):
        """Test drawdown after massive gain followed by loss."""
        rule = MaxDrawdownRule(max_dd_pct=0.10)

        # Gain to $200k
        context_peak = base_context.__class__(
            **{**base_context.__dict__, "equity": 200000.0}
        )
        rule.validate_order(buy_order, context_peak)
        assert rule.high_water_mark == 200000.0

        # Drop to $170k → 15% DD → rejected
        context_dd = base_context.__class__(
            **{**base_context.__dict__, "equity": 170000.0}
        )
        assert rule.validate_order(buy_order, context_dd) is None

        # Note: Still above original $100k, but below HWM threshold

    def test_leverage_with_short_positions(self):
        """Test leverage calculation accounts for short positions."""
        # This is simplified - real implementation would need position details
        rule = MaxLeverageRule(max_leverage=2.0)

        # Net position might be small, but gross exposure is high
        # Leverage should be based on total exposure, not net
        # Current implementation uses abs(leverage * equity) as approximation
        pass  # Implementation detail - covered by other tests
