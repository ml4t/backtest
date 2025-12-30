"""Unit tests for the Position class."""

from datetime import datetime

import pytest

from ml4t.backtest import Position


class TestPositionLongPositions:
    """Tests for long position scenarios."""

    def test_long_position_creation(self):
        """Test creating a basic long position."""
        pos = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=150.0,
            current_price=150.0,
            entry_time=datetime(2025, 1, 1, 10, 0),
            bars_held=0,
        )

        assert pos.asset == "AAPL"
        assert pos.quantity == 100.0
        assert pos.avg_entry_price == 150.0
        assert pos.current_price == 150.0
        assert pos.bars_held == 0

    def test_long_position_market_value(self):
        """Test market value calculation for long position."""
        pos = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=150.0,
            current_price=155.0,
            entry_time=datetime.now(),
        )

        assert pos.market_value == 15500.0  # 100 * 155

    def test_long_position_profit(self):
        """Test unrealized P&L for profitable long position."""
        pos = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=150.0,
            current_price=155.0,
            entry_time=datetime.now(),
        )

        assert pos.unrealized_pnl() == 500.0  # (155 - 150) * 100

    def test_long_position_loss(self):
        """Test unrealized P&L for losing long position."""
        pos = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=150.0,
            current_price=145.0,
            entry_time=datetime.now(),
        )

        assert pos.unrealized_pnl() == -500.0  # (145 - 150) * 100

    def test_long_position_zero_pnl(self):
        """Test unrealized P&L when price hasn't changed."""
        pos = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=150.0,
            current_price=150.0,
            entry_time=datetime.now(),
        )

        assert pos.unrealized_pnl() == 0.0

    def test_long_position_fractional_quantity(self):
        """Test position with fractional shares."""
        pos = Position(
            asset="BTC-USD",
            quantity=0.5,
            entry_price=50000.0,
            current_price=55000.0,
            entry_time=datetime.now(),
        )

        assert pos.market_value == 27500.0  # 0.5 * 55000
        assert pos.unrealized_pnl() == 2500.0  # (55000 - 50000) * 0.5


class TestPositionShortPositions:
    """Tests for short position scenarios."""

    def test_short_position_creation(self):
        """Test creating a short position (negative quantity)."""
        pos = Position(
            asset="AAPL",
            quantity=-100.0,
            entry_price=150.0,
            current_price=150.0,
            entry_time=datetime(2025, 1, 1, 10, 0),
        )

        assert pos.quantity == -100.0
        assert pos.avg_entry_price == 150.0

    def test_short_position_market_value_is_negative(self):
        """Test that short positions have negative market value (liability)."""
        pos = Position(
            asset="AAPL",
            quantity=-100.0,
            entry_price=150.0,
            current_price=155.0,
            entry_time=datetime.now(),
        )

        # Short position = liability on balance sheet
        assert pos.market_value == -15500.0  # -100 * 155

    def test_short_position_profit_when_price_drops(self):
        """Test short position profits when price decreases."""
        pos = Position(
            asset="AAPL",
            quantity=-100.0,
            entry_price=150.0,
            current_price=145.0,  # Price dropped
            entry_time=datetime.now(),
        )

        # Sold at 150, now at 145 -> profit
        assert pos.unrealized_pnl() == 500.0  # (145 - 150) * (-100)

    def test_short_position_loss_when_price_rises(self):
        """Test short position loses when price increases."""
        pos = Position(
            asset="AAPL",
            quantity=-100.0,
            entry_price=150.0,
            current_price=155.0,  # Price rose
            entry_time=datetime.now(),
        )

        # Sold at 150, now at 155 -> loss
        assert pos.unrealized_pnl() == -500.0  # (155 - 150) * (-100)

    def test_short_position_zero_pnl(self):
        """Test short position with no price change."""
        pos = Position(
            asset="AAPL",
            quantity=-100.0,
            entry_price=150.0,
            current_price=150.0,
            entry_time=datetime.now(),
        )

        assert pos.unrealized_pnl() == 0.0


class TestPositionEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_quantity_position(self):
        """Test position with zero quantity (flat)."""
        pos = Position(
            asset="AAPL",
            quantity=0.0,
            entry_price=150.0,
            current_price=155.0,
            entry_time=datetime.now(),
        )

        assert pos.market_value == 0.0
        assert pos.unrealized_pnl() == 0.0

    def test_very_small_quantity(self):
        """Test position with very small fractional quantity."""
        pos = Position(
            asset="ETH-USD",
            quantity=0.001,
            entry_price=3000.0,
            current_price=3100.0,
            entry_time=datetime.now(),
        )

        assert pos.market_value == pytest.approx(3.1, abs=0.01)
        assert pos.unrealized_pnl() == pytest.approx(0.1, abs=0.01)

    def test_large_price_change(self):
        """Test position with very large price change."""
        pos = Position(
            asset="MEME",
            quantity=1000.0,
            entry_price=1.0,
            current_price=100.0,  # 100x gain
            entry_time=datetime.now(),
        )

        assert pos.market_value == 100000.0
        assert pos.unrealized_pnl() == 99000.0

    def test_bars_held_tracking(self):
        """Test that bars_held is tracked correctly."""
        pos = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=150.0,
            current_price=150.0,
            entry_time=datetime.now(),
            bars_held=42,
        )

        assert pos.bars_held == 42

    def test_bars_held_defaults_to_zero(self):
        """Test that bars_held defaults to 0 if not provided."""
        pos = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=150.0,
            current_price=150.0,
            entry_time=datetime.now(),
        )

        assert pos.bars_held == 0


class TestPositionRepresentation:
    """Tests for Position string representation."""

    def test_long_position_repr(self):
        """Test __repr__ for long position."""
        pos = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=150.0,
            current_price=155.0,
            entry_time=datetime.now(),
        )

        repr_str = repr(pos)
        assert "LONG" in repr_str
        assert "100.00" in repr_str
        assert "AAPL" in repr_str
        assert "$150.00" in repr_str
        assert "$155.00" in repr_str
        assert "+500.00" in repr_str  # Positive P&L

    def test_short_position_repr(self):
        """Test __repr__ for short position."""
        pos = Position(
            asset="AAPL",
            quantity=-100.0,
            entry_price=150.0,
            current_price=145.0,
            entry_time=datetime.now(),
        )

        repr_str = repr(pos)
        assert "SHORT" in repr_str
        assert "100.00" in repr_str  # Absolute value
        assert "AAPL" in repr_str
        assert "+500.00" in repr_str  # Profit from price drop

    def test_zero_position_repr(self):
        """Test __repr__ for zero position."""
        pos = Position(
            asset="AAPL",
            quantity=0.0,
            entry_price=150.0,
            current_price=150.0,
            entry_time=datetime.now(),
        )

        repr_str = repr(pos)
        # Zero quantity should show SHORT (since quantity < 0 is false, goes to else)
        assert "SHORT" in repr_str or "LONG" in repr_str  # Implementation detail


class TestPositionMarkToMarket:
    """Tests for mark-to-market price updates."""

    def test_mark_to_market_updates_market_value(self):
        """Test that changing current_price updates market_value."""
        pos = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=150.0,
            current_price=150.0,
            entry_time=datetime.now(),
        )

        assert pos.market_value == 15000.0

        # Simulate mark-to-market update
        pos.current_price = 160.0
        assert pos.market_value == 16000.0

    def test_mark_to_market_updates_unrealized_pnl(self):
        """Test that changing current_price updates unrealized P&L."""
        pos = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=150.0,
            current_price=150.0,
            entry_time=datetime.now(),
        )

        assert pos.unrealized_pnl() == 0.0

        # Simulate mark-to-market update
        pos.current_price = 160.0
        assert pos.unrealized_pnl() == 1000.0  # (160 - 150) * 100


class TestPositionCostBasisTracking:
    """Tests for weighted average cost basis tracking."""

    def test_avg_entry_price_represents_cost_basis(self):
        """Test that avg_entry_price is the cost basis."""
        # This test documents that entry_price (via avg_entry_price property)
        # is used for cost basis. Updates are done externally
        # (e.g., by AccountState when adding to a position)

        pos = Position(
            asset="AAPL",
            quantity=100.0,
            entry_price=150.0,
            current_price=150.0,
            entry_time=datetime.now(),
        )

        # Verify cost basis is tracked (avg_entry_price is alias for entry_price)
        assert pos.avg_entry_price == 150.0
        assert pos.entry_price == 150.0

        # Simulating adding 100 shares at $160
        # Weighted average: (100*150 + 100*160) / 200 = 155
        # This would be calculated externally and set via entry_price:
        pos.entry_price = 155.0
        pos.quantity = 200.0

        # Verify avg_entry_price property reflects updated entry_price
        assert pos.avg_entry_price == 155.0

        # Now verify P&L is based on weighted average
        pos.current_price = 160.0
        assert pos.unrealized_pnl() == 1000.0  # (160 - 155) * 200


class TestPositionRealWorldScenarios:
    """Tests using realistic trading scenarios."""

    def test_day_trader_scenario(self):
        """Test scenario: Day trader buys TSLA, price moves, sells."""
        # Buy 50 shares at $200
        pos = Position(
            asset="TSLA",
            quantity=50.0,
            entry_price=200.0,
            current_price=200.0,
            entry_time=datetime(2025, 1, 15, 9, 30),
            bars_held=0,
        )

        # Price rises to $205
        pos.current_price = 205.0
        pos.bars_held = 5

        assert pos.market_value == 10250.0
        assert pos.unrealized_pnl() == 250.0  # $5 * 50 shares

    def test_crypto_fractional_shares(self):
        """Test scenario: Buying fractional BTC."""
        # Buy 0.1 BTC at $60,000
        pos = Position(
            asset="BTC-USD",
            quantity=0.1,
            entry_price=60000.0,
            current_price=60000.0,
            entry_time=datetime.now(),
        )

        # Price drops to $55,000
        pos.current_price = 55000.0

        assert pos.market_value == 5500.0
        assert pos.unrealized_pnl() == -500.0  # ($55k - $60k) * 0.1

    def test_short_seller_scenario(self):
        """Test scenario: Short selling overvalued meme stock."""
        # Short 200 shares at $50
        pos = Position(
            asset="MEME",
            quantity=-200.0,
            entry_price=50.0,
            current_price=50.0,
            entry_time=datetime.now(),
        )

        # Price drops to $30 (short profits!)
        pos.current_price = 30.0

        assert pos.market_value == -6000.0  # Short = liability
        assert pos.unrealized_pnl() == 4000.0  # (30 - 50) * (-200)

        # Price rises to $60 (short loses)
        pos.current_price = 60.0

        assert pos.market_value == -12000.0
        assert pos.unrealized_pnl() == -2000.0  # (60 - 50) * (-200)
