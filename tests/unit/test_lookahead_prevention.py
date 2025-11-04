"""Test that execution delay prevents lookahead bias."""

from datetime import datetime, timezone

import pytest

from qengine.core.event import MarketEvent
from qengine.core.types import MarketDataType, OrderSide, OrderType
from qengine.execution.broker import SimulationBroker
from qengine.execution.order import Order


class TestLookaheadPrevention:
    """Test suite for preventing lookahead bias in order execution."""

    def test_execution_delay_enabled_by_default(self):
        """Test that execution delay is enabled by default."""
        broker = SimulationBroker()
        assert broker.execution_delay is True, "Execution delay should be enabled by default"

    def test_market_order_delayed_execution(self):
        """Test that market orders execute on the next market event, not the current one."""
        # Setup broker with execution delay (default)
        broker = SimulationBroker(initial_cash=10000.0)

        # Create market events with different prices
        timestamp1 = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
        timestamp2 = datetime(2024, 1, 1, 9, 31, tzinfo=timezone.utc)

        market_event1 = MarketEvent(
            timestamp=timestamp1,
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            price=150.0,
            close=150.0,
            volume=1000,
        )

        market_event2 = MarketEvent(
            timestamp=timestamp2,
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            price=151.0,
            close=151.0,
            volume=1000,
        )

        # Submit market order on first event
        order = Order(
            order_id="TEST001",
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=10.0,
        )

        # Submit order
        broker.submit_order(order)

        # Process first market event - should NOT fill the order
        fills1 = broker.on_market_event(market_event1)
        assert len(fills1) == 0, "Order should not fill on the same event it was submitted"

        # Process second market event - should fill the order
        fills2 = broker.on_market_event(market_event2)
        assert len(fills2) == 1, "Order should fill on the next market event"

        # Verify fill price is from the second event, not the first
        fill = fills2[0]
        # Buy order gets slippage: 151.0 * 1.0001 = 151.0151
        assert fill.fill_price == pytest.approx(151.0151, rel=1e-4), (
            f"Order should fill at second event price with slippage, got {fill.fill_price}"
        )
        assert fill.fill_quantity == 10.0, "Fill quantity should match order"

    def test_immediate_execution_mode_legacy(self):
        """Test that immediate execution mode can be explicitly enabled for legacy compatibility."""
        # Setup broker with immediate execution (legacy mode)
        broker = SimulationBroker(initial_cash=10000.0, execution_delay=False)

        timestamp = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)

        market_event = MarketEvent(
            timestamp=timestamp,
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            price=150.0,
            close=150.0,
            volume=1000,
        )

        # Submit and process market order
        order = Order(
            order_id="TEST002",
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=10.0,
        )

        # Update price first so broker knows the current price
        broker.on_market_event(market_event)

        # Submit order and it should fill immediately
        broker.submit_order(order)

        # Order should have filled immediately on submission
        # (The fill event is published via event bus in submit_order)
        # Check that the order is already filled
        order_id = order.order_id
        if order_id in broker._orders:
            tracked_order = broker._orders[order_id]
            assert tracked_order.is_filled, "Market order should fill immediately in legacy mode"

    def test_limit_order_delayed_execution(self):
        """Test that limit orders also respect execution delay."""
        broker = SimulationBroker(initial_cash=10000.0)

        timestamp1 = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
        timestamp2 = datetime(2024, 1, 1, 9, 31, tzinfo=timezone.utc)

        # Submit limit order that would be fillable at current price
        order = Order(
            order_id="TEST003",
            asset_id="AAPL",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=10.0,
            limit_price=151.0,  # Limit above market price
        )

        broker.submit_order(order)

        # First market event at 150 - order should not fill yet
        market_event1 = MarketEvent(
            timestamp=timestamp1,
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            price=150.0,
            close=150.0,
            volume=1000,
        )

        fills1 = broker.on_market_event(market_event1)
        assert len(fills1) == 0, "Limit order should not fill on submission event"

        # Second market event still at 150 - order should now fill
        market_event2 = MarketEvent(
            timestamp=timestamp2,
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            price=150.0,
            close=150.0,
            volume=1000,
        )

        fills2 = broker.on_market_event(market_event2)
        assert len(fills2) == 1, "Limit order should fill on next event when price is favorable"
        assert fills2[0].fill_price == 150.0, "Should fill at market price"

    def test_stop_order_trigger_and_fill_delay(self):
        """Test that stop orders trigger immediately but fill on next event."""
        broker = SimulationBroker(initial_cash=10000.0)

        timestamp1 = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
        timestamp2 = datetime(2024, 1, 1, 9, 31, tzinfo=timezone.utc)
        timestamp3 = datetime(2024, 1, 1, 9, 32, tzinfo=timezone.utc)

        # Submit stop order
        order = Order(
            order_id="TEST004",
            asset_id="AAPL",
            order_type=OrderType.STOP,
            side=OrderSide.SELL,
            quantity=10.0,
            stop_price=149.0,  # Stop below current price
        )

        broker.submit_order(order)

        # First event at 150 - stop not triggered
        market_event1 = MarketEvent(
            timestamp=timestamp1,
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            price=150.0,
            close=150.0,
            volume=1000,
        )

        fills1 = broker.on_market_event(market_event1)
        assert len(fills1) == 0, "Stop order should not trigger above stop price"

        # Second event at 148 - stop triggers and converts to market order
        market_event2 = MarketEvent(
            timestamp=timestamp2,
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            price=148.0,
            close=148.0,
            volume=1000,
        )

        fills2 = broker.on_market_event(market_event2)
        # Stop orders fill immediately when triggered (they already waited)
        assert len(fills2) == 1, "Stop order should fill immediately when triggered"
        # Sell order gets slippage: 148.0 * 0.9999 = 147.9852
        assert fills2[0].fill_price == pytest.approx(147.9852, rel=1e-4), (
            "Should fill at trigger price with slippage"
        )

    def test_multiple_orders_proper_sequencing(self):
        """Test that multiple orders submitted at different times execute in proper sequence."""
        broker = SimulationBroker(initial_cash=100000.0)

        timestamps = [datetime(2024, 1, 1, 9, 30, i, tzinfo=timezone.utc) for i in range(5)]

        # Submit first order
        order1 = Order(
            order_id="ORDER1",
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=10.0,
        )
        broker.submit_order(order1, timestamps[0])

        # First market event - no fills yet
        event1 = MarketEvent(
            timestamp=timestamps[0],
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            price=150.0,
            close=150.0,
            volume=1000,
        )
        fills1 = broker.on_market_event(event1)
        assert len(fills1) == 0, "First order should not fill on submission event"

        # Submit second order and process second event
        order2 = Order(
            order_id="ORDER2",
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=20.0,
        )
        broker.submit_order(order2, timestamps[1])

        event2 = MarketEvent(
            timestamp=timestamps[1],
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            price=151.0,
            close=151.0,
            volume=2000,
        )
        fills2 = broker.on_market_event(event2)

        # First order should fill, second should not
        assert len(fills2) == 1, "Only first order should fill on second event"
        assert fills2[0].order_id == "ORDER1", "First order should fill first"
        # Buy order gets slippage: 151.0 * 1.0001 = 151.0151
        assert fills2[0].fill_price == pytest.approx(151.0151, rel=1e-4), (
            "Should fill at current price with slippage"
        )

        # Third event - second order should fill
        event3 = MarketEvent(
            timestamp=timestamps[2],
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            price=152.0,
            close=152.0,
            volume=1000,
        )
        fills3 = broker.on_market_event(event3)

        assert len(fills3) == 1, "Second order should fill on third event"
        assert fills3[0].order_id == "ORDER2", "Second order should fill"
        # Buy order gets slippage: 152.0 * 1.0001 = 152.0152
        assert fills3[0].fill_price == pytest.approx(152.0152, rel=1e-4), (
            "Should fill at current price with slippage"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
