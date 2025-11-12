"""Unit tests for PositionTracker - Phase 1 validation."""

import pytest

from qengine.portfolio.core import PositionTracker


class TestPositionTrackerCore:
    """Test core PositionTracker functionality."""

    def test_initialization(self):
        """Test PositionTracker initialization."""
        tracker = PositionTracker(initial_cash=100000.0)

        assert tracker.initial_cash == 100000.0
        assert tracker.cash == 100000.0
        assert len(tracker.positions) == 0
        assert tracker.equity == 100000.0
        assert tracker.total_commission == 0.0
        assert tracker.total_slippage == 0.0
        assert tracker.total_realized_pnl == 0.0

    def test_buy_position(self):
        """Test buying a position."""
        tracker = PositionTracker(initial_cash=100000.0)

        # Buy 10 shares at $50
        tracker.update_position(
            asset_id="AAPL",
            quantity_change=10.0,
            price=50.0,
            commission=5.0,
        )

        assert tracker.cash == 99495.0  # 100000 - (10*50 + 5)
        assert len(tracker.positions) == 1
        assert tracker.get_position("AAPL") is not None
        assert tracker.get_position("AAPL").quantity == 10.0
        assert tracker.get_position("AAPL").cost_basis == 500.0
        assert tracker.total_commission == 5.0

    def test_sell_position(self):
        """Test selling a position."""
        tracker = PositionTracker(initial_cash=100000.0)

        # Buy 10 shares at $50
        tracker.update_position("AAPL", quantity_change=10.0, price=50.0, commission=5.0)

        # Sell 10 shares at $60
        tracker.update_position("AAPL", quantity_change=-10.0, price=60.0, commission=5.0)

        # Cash should be: 100000 - 505 (buy) + 595 (sell) = 100090
        assert tracker.cash == 100090.0
        assert len(tracker.positions) == 0  # Position closed
        assert tracker.total_realized_pnl == 100.0  # (60-50)*10
        assert tracker.total_commission == 10.0

    def test_partial_sell(self):
        """Test partial position sell."""
        tracker = PositionTracker(initial_cash=100000.0)

        # Buy 10 shares at $50
        tracker.update_position("AAPL", quantity_change=10.0, price=50.0)

        # Sell 5 shares at $60
        tracker.update_position("AAPL", quantity_change=-5.0, price=60.0)

        assert tracker.get_position("AAPL").quantity == 5.0
        assert tracker.total_realized_pnl == 50.0  # (60-50)*5

    def test_update_prices(self):
        """Test updating market prices."""
        tracker = PositionTracker(initial_cash=100000.0)

        # Buy 10 shares at $50
        tracker.update_position("AAPL", quantity_change=10.0, price=50.0)

        # Update market price to $60
        tracker.update_prices({"AAPL": 60.0})

        position = tracker.get_position("AAPL")
        assert position.last_price == 60.0
        assert position.market_value == 600.0  # 10 * 60
        assert position.unrealized_pnl == 100.0  # (60-50)*10

    def test_equity_calculation(self):
        """Test equity calculation."""
        tracker = PositionTracker(initial_cash=100000.0)

        # Buy 10 shares at $50
        tracker.update_position("AAPL", quantity_change=10.0, price=50.0)

        # Update market price to $60
        tracker.update_prices({"AAPL": 60.0})

        # Equity = cash + position value
        # Cash = 100000 - 500 = 99500
        # Position value = 10 * 60 = 600
        assert tracker.equity == 100100.0

    def test_returns_calculation(self):
        """Test returns calculation."""
        tracker = PositionTracker(initial_cash=100000.0)

        # Buy and price up
        tracker.update_position("AAPL", quantity_change=10.0, price=50.0)
        tracker.update_prices({"AAPL": 60.0})

        # Returns = (equity - initial) / initial
        # Equity = 100100, Initial = 100000
        assert tracker.returns == 0.001  # 0.1%

    def test_multiple_positions(self):
        """Test tracking multiple positions."""
        tracker = PositionTracker(initial_cash=100000.0)

        # Buy multiple assets
        tracker.update_position("AAPL", quantity_change=10.0, price=50.0)
        tracker.update_position("GOOGL", quantity_change=5.0, price=100.0)

        assert len(tracker.positions) == 2
        assert tracker.get_position("AAPL").quantity == 10.0
        assert tracker.get_position("GOOGL").quantity == 5.0

        # Cash = 100000 - 500 - 500 = 99000
        assert tracker.cash == 99000.0

    def test_get_summary(self):
        """Test get_summary method."""
        tracker = PositionTracker(initial_cash=100000.0)

        # Buy and price up
        tracker.update_position("AAPL", quantity_change=10.0, price=50.0, commission=5.0)
        tracker.update_prices({"AAPL": 60.0})

        summary = tracker.get_summary()

        assert summary["cash"] == 99495.0
        assert summary["equity"] == 100095.0
        assert summary["positions"] == 1
        assert summary["unrealized_pnl"] == 100.0
        assert summary["commission"] == 5.0
        assert "returns" in summary

    def test_reset(self):
        """Test reset functionality."""
        tracker = PositionTracker(initial_cash=100000.0)

        # Buy something
        tracker.update_position("AAPL", quantity_change=10.0, price=50.0, commission=5.0)

        # Reset
        tracker.reset()

        assert tracker.cash == 100000.0
        assert len(tracker.positions) == 0
        assert tracker.total_commission == 0.0
        assert tracker.total_realized_pnl == 0.0

    def test_per_asset_realized_pnl(self):
        """Test per-asset P&L tracking."""
        tracker = PositionTracker(initial_cash=100000.0)

        # Trade AAPL
        tracker.update_position("AAPL", quantity_change=10.0, price=50.0)
        tracker.update_position("AAPL", quantity_change=-10.0, price=60.0)

        # Trade GOOGL
        tracker.update_position("GOOGL", quantity_change=5.0, price=100.0)
        tracker.update_position("GOOGL", quantity_change=-5.0, price=110.0)

        assert tracker.asset_realized_pnl["AAPL"] == 100.0
        assert tracker.asset_realized_pnl["GOOGL"] == 50.0
        assert tracker.total_realized_pnl == 150.0


class TestPositionTrackerEdgeCases:
    """Test edge cases and error conditions."""

    def test_cannot_sell_more_than_owned(self):
        """Test that selling more than owned raises error."""
        tracker = PositionTracker(initial_cash=100000.0)

        # Buy 10 shares
        tracker.update_position("AAPL", quantity_change=10.0, price=50.0)

        # Try to sell 15 shares - should raise error
        with pytest.raises(ValueError, match="Cannot remove"):
            tracker.update_position("AAPL", quantity_change=-15.0, price=60.0)

    def test_zero_initial_cash(self):
        """Test tracker with zero initial cash."""
        tracker = PositionTracker(initial_cash=0.0)

        assert tracker.initial_cash == 0.0
        assert tracker.equity == 0.0
        assert tracker.returns == 0.0  # Should handle division by zero

    def test_get_nonexistent_position(self):
        """Test getting a position that doesn't exist."""
        tracker = PositionTracker(initial_cash=100000.0)

        assert tracker.get_position("NONEXISTENT") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
