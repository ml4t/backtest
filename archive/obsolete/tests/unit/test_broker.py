"""Unit tests for broker implementations."""

from datetime import datetime

import pytest

from ml4t.backtest.core.event import MarketEvent
from ml4t.backtest.core.types import MarketDataType, OrderSide, OrderType
from ml4t.backtest.execution.broker import SimulationBroker
from ml4t.backtest.execution.order import Order, OrderState


class TestSimulationBroker:
    """Test suite for SimulationBroker."""

    @pytest.fixture
    def broker(self):
        """Create a broker instance for testing."""
        # Use immediate execution for basic tests
        return SimulationBroker(initial_cash=10000.0, execution_delay=False)

    @pytest.fixture
    def market_order(self):
        """Create a sample market order."""
        return Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100.0,
        )

    @pytest.fixture
    def limit_order(self):
        """Create a sample limit order."""
        return Order(
            asset_id="AAPL",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=100.0,
            limit_price=150.0,
        )

    def test_broker_initialization(self):
        """Test broker initialization."""
        broker = SimulationBroker(initial_cash=50000.0)
        assert broker.cash == 50000.0
        assert broker.get_positions() == {}
        assert broker.get_open_orders() == []

    def test_submit_market_order(self, broker, market_order):
        """Test submitting a market order."""
        order_id = broker.submit_order(market_order)

        assert order_id == market_order.order_id
        assert market_order.state == OrderState.SUBMITTED

        # Check order is tracked
        retrieved_order = broker.get_order(order_id)
        assert retrieved_order == market_order

        # Check order is in open orders
        open_orders = broker.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0] == market_order

    def test_submit_limit_order(self, broker, limit_order):
        """Test submitting a limit order."""
        order_id = broker.submit_order(limit_order)

        assert order_id == limit_order.order_id
        assert limit_order.state == OrderState.SUBMITTED

        open_orders = broker.get_open_orders("AAPL")
        assert len(open_orders) == 1
        assert open_orders[0] == limit_order

    def test_cancel_order(self, broker, market_order):
        """Test cancelling an order."""
        order_id = broker.submit_order(market_order)

        # Cancel the order
        success = broker.cancel_order(order_id)
        assert success is True
        assert market_order.state == OrderState.CANCELLED

        # Check order is no longer in open orders
        open_orders = broker.get_open_orders()
        assert len(open_orders) == 0

        # Try to cancel again - should fail
        success = broker.cancel_order(order_id)
        assert success is False

    def test_market_order_fill(self, broker):
        """Test market order execution with sufficient cash."""
        # Use smaller order that fits within cash balance
        small_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=50.0,
        )
        broker.submit_order(small_order)

        # Create market event
        market_event = MarketEvent(
            timestamp=datetime.now(),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=150.0,
        )

        # Process market event
        fills = broker.on_market_event(market_event)

        assert len(fills) == 1
        fill = fills[0]

        # Check fill details
        assert fill.order_id == small_order.order_id
        assert fill.asset_id == "AAPL"
        assert fill.side == OrderSide.BUY
        assert fill.fill_quantity == 50.0
        assert fill.fill_price == pytest.approx(150.0 * 1.0001)  # With slippage

        # Check order is filled
        assert small_order.is_filled
        assert small_order.filled_quantity == 50.0

        # Check position updated
        assert broker.get_position("AAPL") == 50.0

        # Check cash updated (price + commission)
        expected_cash = 10000.0 - (50.0 * fill.fill_price + 1.0)
        assert broker.cash == pytest.approx(expected_cash)

    def test_limit_order_fill_when_price_favorable(self, broker):
        """Test limit order fills when price is favorable."""
        # Use smaller order that fits within cash balance
        small_limit = Order(
            asset_id="AAPL",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=50.0,
            limit_price=150.0,
        )
        broker.submit_order(small_limit)

        # Price below limit - should fill
        market_event = MarketEvent(
            timestamp=datetime.now(),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=149.0,
        )

        fills = broker.on_market_event(market_event)

        assert len(fills) == 1
        assert small_limit.is_filled
        assert fills[0].fill_price <= 150.0

    def test_limit_order_no_fill_when_price_unfavorable(self, broker, limit_order):
        """Test limit order doesn't fill when price is unfavorable."""
        broker.submit_order(limit_order)

        # Price above limit - should not fill
        market_event = MarketEvent(
            timestamp=datetime.now(),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=151.0,
        )

        fills = broker.on_market_event(market_event)

        assert len(fills) == 0
        assert not limit_order.is_filled
        assert limit_order.is_active

    def test_sell_order_with_position(self, broker):
        """Test selling when we have a position."""
        # First buy some shares
        buy_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100.0,
        )
        broker.submit_order(buy_order)

        market_event = MarketEvent(
            timestamp=datetime.now(),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=150.0,
        )
        broker.on_market_event(market_event)

        # Now sell
        sell_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=50.0,
        )
        broker.submit_order(sell_order)

        market_event = MarketEvent(
            timestamp=datetime.now(),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=151.0,
        )
        fills = broker.on_market_event(market_event)

        assert len(fills) == 1
        assert sell_order.is_filled
        assert broker.get_position("AAPL") == 50.0  # 100 - 50

    def test_sell_order_without_position(self, broker):
        """Test selling when we don't have a position (should not fill)."""
        sell_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=100.0,
        )
        broker.submit_order(sell_order)

        market_event = MarketEvent(
            timestamp=datetime.now(),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=150.0,
        )
        fills = broker.on_market_event(market_event)

        assert len(fills) == 0
        assert not sell_order.is_filled

    def test_stop_order_trigger(self, broker):
        """Test stop order triggering."""
        # First, get a position with smaller order
        buy_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=50.0,
        )
        broker.submit_order(buy_order)

        market_event = MarketEvent(
            timestamp=datetime.now(),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=150.0,
        )
        broker.on_market_event(market_event)

        # Now create stop order to sell the position
        stop_order = Order(
            asset_id="AAPL",
            order_type=OrderType.STOP,
            side=OrderSide.SELL,
            quantity=50.0,
            stop_price=145.0,
        )
        broker.submit_order(stop_order)

        # Price above stop - should not trigger
        market_event = MarketEvent(
            timestamp=datetime.now(),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=146.0,
        )
        fills = broker.on_market_event(market_event)
        assert len(fills) == 0

        # Price at/below stop - should trigger and fill
        market_event = MarketEvent(
            timestamp=datetime.now(),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=144.0,
        )
        fills = broker.on_market_event(market_event)

        assert len(fills) == 1
        assert stop_order.is_filled
        assert broker.get_position("AAPL") == 0.0

    def test_insufficient_cash(self, broker):
        """Test order rejection when insufficient cash."""
        # Try to buy more than we can afford
        large_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1000.0,  # Would cost ~$150,000
        )
        broker.submit_order(large_order)

        market_event = MarketEvent(
            timestamp=datetime.now(),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=150.0,
        )
        fills = broker.on_market_event(market_event)

        # Should get partial fill up to available cash
        assert len(fills) == 1
        assert large_order.is_partially_filled
        assert large_order.filled_quantity < 1000.0

        # Cash should be nearly depleted
        assert broker.cash < 100.0

    def test_multiple_orders_same_asset(self, broker):
        """Test multiple orders for the same asset."""
        order1 = Order(
            asset_id="AAPL",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=30.0,  # Smaller to fit in cash
            limit_price=149.0,  # Will NOT fill at 150
        )
        order2 = Order(
            asset_id="AAPL",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=40.0,  # Smaller to fit in cash
            limit_price=150.0,  # Will fill at 150
        )

        broker.submit_order(order1)
        broker.submit_order(order2)

        # Price at 150 - only order2 should fill
        market_event = MarketEvent(
            timestamp=datetime.now(),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=150.0,
        )
        fills = broker.on_market_event(market_event)

        assert len(fills) == 1
        assert fills[0].order_id == order2.order_id
        assert order2.is_filled
        assert not order1.is_filled

    def test_broker_statistics(self, broker):
        """Test broker statistics tracking."""
        # Execute some trades with smaller order
        order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=50.0,
        )
        broker.submit_order(order)

        market_event = MarketEvent(
            timestamp=datetime.now(),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=150.0,
        )
        broker.on_market_event(market_event)

        stats = broker.get_statistics()

        assert stats["fill_count"] == 1
        assert stats["total_commission"] == 1.0
        assert stats["total_slippage"] > 0
        # After filling, the order should be removed from open orders
        assert stats["open_orders"] == 0
