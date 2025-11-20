"""Targeted tests to achieve 90%+ coverage on portfolio module.

This test suite covers edge cases and less-common code paths to ensure
comprehensive coverage of the new Portfolio architecture.

Coverage targets:
- state.py: 85% → 90%+ (PrecisionManager edge cases, PortfolioState metrics)
- portfolio.py: 90% → 92%+ (reset with analyzer)
- core.py: 91% → 92%+ (PrecisionManager rounding operations)
"""

import pytest
from datetime import datetime
from decimal import Decimal

from ml4t.backtest.core.event import FillEvent, OrderSide
from ml4t.backtest.core.precision import PrecisionManager
from ml4t.backtest.portfolio.portfolio import Portfolio
from ml4t.backtest.portfolio.core import PositionTracker
from ml4t.backtest.portfolio.state import Position, PortfolioState


# ===== Tests for state.py =====


class TestPositionWithPrecisionManager:
    """Test Position with PrecisionManager edge cases."""

    @pytest.fixture
    def precision_manager(self):
        """Create precision manager."""
        return PrecisionManager(position_decimals=8, price_decimals=2, cash_decimals=2)

    @pytest.fixture
    def position(self, precision_manager):
        """Create position with precision manager."""
        return Position(
            asset_id="BTC",
            precision_manager=precision_manager,
        )

    def test_update_price_with_precision_rounding(self, position, precision_manager):
        """Test that update_price rounds unrealized P&L with precision manager."""
        position.add_shares(1.12345678, 50000.0)
        position.update_price(51000.0)

        # unrealized_pnl should be rounded to 2 decimal places
        expected_pnl = 1.12345678 * (51000.0 - 50000.0)
        rounded_pnl = precision_manager.round_cash(expected_pnl)
        assert position.unrealized_pnl == rounded_pnl

    def test_add_shares_with_precision_rounding(self, position, precision_manager):
        """Test that add_shares rounds quantity and cost basis."""
        # Add shares that need rounding
        position.add_shares(1.123456789, 50000.0)

        # Quantity should be rounded to 8 decimals
        expected_qty = precision_manager.round_quantity(1.123456789)
        assert position.quantity == expected_qty

        # Cost basis should be rounded to 2 decimals
        expected_cost = precision_manager.round_cash(1.123456789 * 50000.0)
        assert position.cost_basis == expected_cost

    def test_add_shares_closing_position_with_precision(self, position, precision_manager):
        """Test closing position detects zero with precision manager."""
        # Open position
        position.add_shares(1.0, 50000.0)

        # Close with very small residual that should be treated as zero
        position.add_shares(-1.0 + 1e-10, 51000.0)

        # Position should be closed (precision-aware zero check)
        assert position.quantity == 0.0
        assert position.cost_basis == 0.0
        assert position.unrealized_pnl == 0.0

    def test_remove_shares_with_precision_rounding(self, position, precision_manager):
        """Test that remove_shares rounds realized P&L, cost basis, and quantity."""
        # Open position
        position.add_shares(2.0, 50000.0)

        # Remove shares
        realized = position.remove_shares(0.5, 52000.0)

        # Realized P&L should be rounded
        expected_realized = 0.5 * (52000.0 - 50000.0)
        assert realized == precision_manager.round_cash(expected_realized)

        # Cost basis should be rounded
        assert position.cost_basis == precision_manager.round_cash(1.5 * 50000.0)

        # Quantity should be rounded
        assert position.quantity == precision_manager.round_quantity(1.5)

    def test_remove_shares_precision_aware_error_check(self, position, precision_manager):
        """Test remove_shares uses precision-aware error checking."""
        position.add_shares(1.0, 50000.0)

        # Try to remove slightly more than available (within precision tolerance)
        # This should work with precision-aware check
        position.remove_shares(1.0 + 1e-10, 51000.0)
        assert position.quantity == pytest.approx(0.0, abs=1e-8)

    def test_remove_shares_without_precision_manager(self):
        """Test remove_shares falls back to fixed tolerance without precision manager."""
        position = Position(asset_id="AAPL")  # No precision manager
        position.add_shares(1.0, 100.0)

        # Should use fixed TOLERANCE = 1e-9
        position.remove_shares(1.0 + 1e-10, 105.0)
        assert position.quantity == pytest.approx(0.0, abs=1e-9)


class TestPortfolioStateMetrics:
    """Test PortfolioState metric calculations."""

    def test_total_pnl_property(self):
        """Test total_pnl combines realized and unrealized P&L."""
        state = PortfolioState(
            timestamp=datetime(2025, 1, 1),
            cash=50000.0,
            total_realized_pnl=1000.0,
            total_unrealized_pnl=500.0,
        )

        assert state.total_pnl == 1500.0

    def test_update_metrics_with_zero_equity(self):
        """Test concentration and leverage calculation when equity is zero or negative."""
        # Test with negative equity (cash negative, no positions)
        state = PortfolioState(
            timestamp=datetime(2025, 1, 1),
            cash=-1000.0,  # Negative cash, no positions = negative equity
        )

        state.update_metrics()

        # When equity <= 0, concentration and leverage should be 0
        assert state.equity == -1000.0
        assert state.concentration == 0.0
        assert state.leverage == 0.0

        # Test with zero equity
        state2 = PortfolioState(
            timestamp=datetime(2025, 1, 1),
            cash=0.0,  # Zero cash, no positions = zero equity
        )

        state2.update_metrics()

        assert state2.equity == 0.0
        assert state2.concentration == 0.0
        assert state2.leverage == 0.0


# ===== Tests for portfolio.py =====


class TestPortfolioReset:
    """Test Portfolio reset() method."""

    def test_reset_with_analyzer_enabled(self):
        """Test reset clears all state including analyzer."""
        portfolio = Portfolio(initial_cash=100000.0, track_analytics=True)

        # Create some state
        fill = FillEvent(
            timestamp=datetime(2025, 1, 1, 10, 0, 0),
            order_id="order_001",
            trade_id="trade_001",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=Decimal("1.0"),
            fill_price=Decimal("50000.0"),
            commission=10.0,
            slippage=5.0,
        )
        portfolio.on_fill_event(fill)
        portfolio.save_state(datetime(2025, 1, 1, 10, 0, 0))

        # Verify state exists
        assert len(portfolio.positions) == 1
        assert len(portfolio.state_history) == 1
        assert portfolio.analyzer is not None
        assert len(portfolio.analyzer.equity_curve) > 0

        # Reset
        portfolio.reset()

        # Verify everything is cleared
        assert portfolio.cash == 100000.0
        assert len(portfolio.positions) == 0
        assert len(portfolio.state_history) == 0
        assert portfolio.total_commission == 0.0
        assert portfolio.total_slippage == 0.0

        # Verify analyzer state is cleared
        assert portfolio.analyzer.high_water_mark == 100000.0
        assert portfolio.analyzer.max_drawdown == 0.0
        assert len(portfolio.analyzer.daily_returns) == 0
        assert len(portfolio.analyzer.timestamps) == 0
        assert len(portfolio.analyzer.equity_curve) == 0
        assert portfolio.analyzer.max_leverage == 0.0
        assert portfolio.analyzer.max_concentration == 0.0

    def test_reset_without_analyzer(self):
        """Test reset works when analytics is disabled."""
        portfolio = Portfolio(initial_cash=100000.0, track_analytics=False)

        # Create some state
        fill = FillEvent(
            timestamp=datetime(2025, 1, 1, 10, 0, 0),
            order_id="order_001",
            trade_id="trade_001",
            asset_id="BTC",
            side=OrderSide.BUY,
            fill_quantity=Decimal("1.0"),
            fill_price=Decimal("50000.0"),
            commission=10.0,
            slippage=5.0,
        )
        portfolio.on_fill_event(fill)

        # Reset
        portfolio.reset()

        # Verify state is cleared
        assert portfolio.cash == 100000.0
        assert len(portfolio.positions) == 0
        assert portfolio.analyzer is None  # Still None after reset


# ===== Tests for core.py =====


class TestPositionTrackerWithPrecisionManager:
    """Test PositionTracker with PrecisionManager rounding."""

    @pytest.fixture
    def precision_manager(self):
        """Create precision manager."""
        return PrecisionManager(position_decimals=8, price_decimals=2, cash_decimals=2)

    @pytest.fixture
    def tracker(self, precision_manager):
        """Create tracker with precision manager."""
        return PositionTracker(
            initial_cash=100000.0,
            precision_manager=precision_manager,
        )

    def test_update_position_buy_rounds_cash(self, tracker, precision_manager):
        """Test that BUY operation rounds cash with precision manager."""
        tracker.update_position(
            asset_id="BTC",
            quantity_change=1.123456789,
            price=50000.12345,
            commission=10.123,
        )

        # Cash should be rounded to 2 decimals
        expected_cash = 100000.0 - (1.123456789 * 50000.12345 + 10.123)
        rounded_cash = precision_manager.round_cash(expected_cash)
        assert tracker.cash == rounded_cash

    def test_update_position_sell_rounds_cash_and_pnl(self, tracker, precision_manager):
        """Test that SELL operation rounds cash and realized P&L."""
        # Buy first
        tracker.update_position(
            asset_id="BTC",
            quantity_change=1.0,
            price=50000.0,
            commission=10.0,
        )

        # Sell
        tracker.update_position(
            asset_id="BTC",
            quantity_change=-0.5,
            price=52000.0,
            commission=10.0,
        )

        # Cash should be rounded to 2 decimals
        expected_cash = (100000.0 - 50000.0 - 10.0) + (0.5 * 52000.0 - 10.0)
        rounded_cash = precision_manager.round_cash(expected_cash)
        assert tracker.cash == rounded_cash

        # Total realized P&L should be rounded
        expected_pnl = 0.5 * (52000.0 - 50000.0)
        rounded_pnl = precision_manager.round_cash(expected_pnl)
        assert tracker.total_realized_pnl == rounded_pnl

    def test_update_position_rounds_asset_realized_pnl(self, tracker, precision_manager):
        """Test that asset-level realized P&L is rounded."""
        # Buy
        tracker.update_position(
            asset_id="BTC",
            quantity_change=1.0,
            price=50000.0,
            commission=10.0,
        )

        # Sell with unusual price to test rounding
        tracker.update_position(
            asset_id="BTC",
            quantity_change=-0.3,
            price=52123.456789,
            commission=5.123,
        )

        # Asset realized P&L should be rounded
        expected_pnl = 0.3 * (52123.456789 - 50000.0)
        rounded_pnl = precision_manager.round_cash(expected_pnl)
        assert tracker.asset_realized_pnl["BTC"] == rounded_pnl

    def test_update_position_rounds_commission_and_slippage(self, tracker, precision_manager):
        """Test that cumulative commission and slippage are rounded."""
        # First trade with unusual values
        tracker.update_position(
            asset_id="BTC",
            quantity_change=1.0,
            price=50000.0,
            commission=10.123456,
            slippage=5.678901,
        )

        # Commission and slippage should be rounded
        assert tracker.total_commission == precision_manager.round_cash(10.123456)
        assert tracker.total_slippage == precision_manager.round_cash(5.678901)

        # Second trade
        tracker.update_position(
            asset_id="ETH",
            quantity_change=10.0,
            price=3000.0,
            commission=8.987654,
            slippage=3.456789,
        )

        # Cumulative values should be rounded
        expected_commission = 10.123456 + 8.987654
        expected_slippage = 5.678901 + 3.456789
        assert tracker.total_commission == precision_manager.round_cash(expected_commission)
        assert tracker.total_slippage == precision_manager.round_cash(expected_slippage)

    def test_position_removal_with_precision_check(self, tracker, precision_manager):
        """Test that empty positions are removed with precision-aware check."""
        # Buy
        tracker.update_position(
            asset_id="BTC",
            quantity_change=1.0,
            price=50000.0,
        )

        assert "BTC" in tracker.positions

        # Sell almost all (leave tiny residual that precision manager treats as zero)
        tracker.update_position(
            asset_id="BTC",
            quantity_change=-1.0 + 1e-10,
            price=51000.0,
        )

        # Position should be removed (precision-aware zero check)
        assert "BTC" not in tracker.positions


# ===== Summary =====
# This test suite adds targeted coverage for:
# 1. PrecisionManager edge cases in Position (state.py)
# 2. PortfolioState metric edge cases (state.py)
# 3. Portfolio.reset() with analyzer (portfolio.py)
# 4. PositionTracker rounding operations (core.py)
#
# Expected coverage improvement:
# - state.py: 85% → 92%+
# - portfolio.py: 90% → 95%+
# - core.py: 91% → 95%+
