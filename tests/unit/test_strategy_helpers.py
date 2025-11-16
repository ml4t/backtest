"""Simplified unit tests for Strategy helper methods.

This test suite focuses on testing the core functionality of helper methods
with realistic broker/portfolio setup.
"""

import pytest
from unittest.mock import Mock, patch

from ml4t.backtest.core.types import OrderSide, OrderType
from ml4t.backtest.execution.broker import SimulationBroker
from ml4t.backtest.portfolio.core import Position
from ml4t.backtest.strategy.base import Strategy


class TestStrategy(Strategy):
    """Test strategy for helper method testing."""

    def on_event(self, event):
        """Required abstract method."""
        pass


@pytest.fixture
def strategy_with_broker():
    """Create a strategy with initialized broker."""
    strategy = TestStrategy()
    broker = SimulationBroker()
    strategy.broker = broker
    return strategy, broker


class TestStrategyHelpers:
    """Test all strategy helper methods."""

    def test_get_position_returns_zero_when_no_position(self, strategy_with_broker):
        """Test get_position returns 0 for non-existent position."""
        strategy, broker = strategy_with_broker
        assert strategy.get_position("AAPL") == 0.0

    def test_get_cash_returns_broker_cash(self, strategy_with_broker):
        """Test get_cash returns cash from broker."""
        strategy, broker = strategy_with_broker
        assert strategy.get_cash() == broker.get_cash()

    def test_get_portfolio_value_returns_equity(self, strategy_with_broker):
        """Test get_portfolio_value returns total equity."""
        strategy, broker = strategy_with_broker
        assert strategy.get_portfolio_value() == broker._internal_portfolio.equity

    def test_buy_percent_submits_order(self, strategy_with_broker):
        """Test buy_percent submits a buy order."""
        strategy, broker = strategy_with_broker

        with patch.object(broker, 'submit_order') as mock_submit:
            strategy.buy_percent("AAPL", 0.10, 150.0)

            # Verify order was submitted
            assert mock_submit.called
            order = mock_submit.call_args[0][0]
            assert order.asset_id == "AAPL"
            assert order.side == OrderSide.BUY
            assert order.quantity > 0

    def test_sell_percent_submits_order_when_position_exists(self, strategy_with_broker):
        """Test sell_percent submits sell order when position exists."""
        strategy, broker = strategy_with_broker

        # Create a position
        position = Position(asset_id="AAPL", quantity=100.0, cost_basis=15000.0, last_price=150.0)
        broker._internal_portfolio.positions["AAPL"] = position

        with patch.object(broker, 'submit_order') as mock_submit:
            strategy.sell_percent("AAPL", 0.50)

            assert mock_submit.called
            order = mock_submit.call_args[0][0]
            assert order.side == OrderSide.SELL
            assert order.quantity == 50  # 50% of 100

    def test_close_position_submits_full_sell(self, strategy_with_broker):
        """Test close_position sells entire position."""
        strategy, broker = strategy_with_broker

        # Create a position
        position = Position(asset_id="AAPL", quantity=100.0, cost_basis=15000.0, last_price=150.0)
        broker._internal_portfolio.positions["AAPL"] = position

        with patch.object(broker, 'submit_order') as mock_submit:
            strategy.close_position("AAPL")

            assert mock_submit.called
            order = mock_submit.call_args[0][0]
            assert order.side == OrderSide.SELL
            assert order.quantity == 100  # Full position

    def test_size_by_confidence_scales_by_confidence(self, strategy_with_broker):
        """Test size_by_confidence scales position size by confidence."""
        strategy, broker = strategy_with_broker

        with patch.object(broker, 'submit_order') as mock_submit:
            # High confidence should use more capital
            strategy.size_by_confidence("AAPL", confidence=0.90, max_percent=0.20, price=150.0)

            assert mock_submit.called
            order = mock_submit.call_args[0][0]
            # 90% of 20% of portfolio
            assert order.quantity > 0

    def test_get_unrealized_pnl_pct_returns_none_when_no_position(self, strategy_with_broker):
        """Test get_unrealized_pnl_pct returns None for non-existent position."""
        strategy, broker = strategy_with_broker
        assert strategy.get_unrealized_pnl_pct("AAPL") is None

    def test_get_unrealized_pnl_pct_calculates_profit(self, strategy_with_broker):
        """Test get_unrealized_pnl_pct calculates profit percentage."""
        strategy, broker = strategy_with_broker

        # Position: bought at $150, now at $165 (+10%)
        position = Position(
            asset_id="AAPL",
            quantity=100.0,
            cost_basis=15000.0,  # 100 shares * $150
            last_price=165.0,
            unrealized_pnl=1500.0,  # (165-150) * 100
        )
        broker._internal_portfolio.positions["AAPL"] = position

        pnl_pct = strategy.get_unrealized_pnl_pct("AAPL")
        assert pnl_pct == pytest.approx(0.10, rel=0.01)  # 10% gain

    def test_rebalance_to_weights_submits_orders(self, strategy_with_broker):
        """Test rebalance_to_weights submits rebalancing orders."""
        strategy, broker = strategy_with_broker

        with patch.object(broker, 'submit_order') as mock_submit:
            target_weights = {"AAPL": 0.40, "GOOGL": 0.30}
            current_prices = {"AAPL": 150.0, "GOOGL": 2800.0}

            strategy.rebalance_to_weights(target_weights, current_prices)

            # Should submit orders for both assets
            assert mock_submit.call_count >= 1


class TestStrategyHelpersNoBroker:
    """Test helper methods raise errors when broker not initialized."""

    def test_get_position_raises_without_broker(self):
        """Test get_position raises error without broker."""
        strategy = TestStrategy()
        with pytest.raises(ValueError, match="Broker not initialized"):
            strategy.get_position("AAPL")

    def test_get_cash_raises_without_broker(self):
        """Test get_cash raises error without broker."""
        strategy = TestStrategy()
        with pytest.raises(ValueError, match="Broker not initialized"):
            strategy.get_cash()

    def test_buy_percent_raises_without_broker(self):
        """Test buy_percent raises error without broker."""
        strategy = TestStrategy()
        with pytest.raises(ValueError, match="Broker not initialized"):
            strategy.buy_percent("AAPL", 0.10, 150.0)
