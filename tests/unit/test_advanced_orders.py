"""Tests for advanced order types in QEngine."""

from datetime import datetime

import pytest

from qengine.core.event import MarketEvent
from qengine.core.types import MarketDataType, OrderSide, OrderType
from qengine.execution.broker import SimulationBroker
from qengine.execution.order import Order, OrderState


class TestStopOrders:
    """Test suite for stop orders."""

    @pytest.fixture
    def broker(self):
        """Create a broker instance."""
        # Use immediate execution for advanced order tests
        return SimulationBroker(initial_cash=100000.0, execution_delay=False)

    def test_stop_order_creation(self):
        """Test stop order validation."""
        order = Order(
            asset_id="AAPL",
            order_type=OrderType.STOP,
            side=OrderSide.SELL,
            quantity=100,
            stop_price=95.0,
        )

        assert order.order_type == OrderType.STOP
        assert order.stop_price == 95.0

    def test_stop_order_missing_price(self):
        """Test stop order without stop price fails."""
        with pytest.raises(ValueError, match="Stop orders must have a stop price"):
            Order(asset_id="AAPL", order_type=OrderType.STOP, side=OrderSide.SELL, quantity=100)

    def test_stop_order_triggering(self, broker):
        """Test stop order triggering mechanism."""
        # First establish a long position
        buy_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )
        broker.submit_order(buy_order)

        # Fill the buy order
        buy_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 9, 30),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=100.0,
        )
        broker.on_market_event(buy_event)

        # Now submit a sell stop order
        order = Order(
            asset_id="AAPL",
            order_type=OrderType.STOP,
            side=OrderSide.SELL,
            quantity=100,
            stop_price=95.0,
        )

        broker.submit_order(order)

        # Price above stop - should not trigger
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=96.0,
        )
        fills = broker.on_market_event(market_event)
        assert len(fills) == 0

        # Price at stop level - should trigger
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 1),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=95.0,
        )
        fills = broker.on_market_event(market_event)
        assert len(fills) == 1
        assert fills[0].asset_id == "AAPL"
        assert fills[0].fill_quantity == 100

    def test_stop_limit_order_creation(self):
        """Test stop-limit order validation."""
        order = Order(
            asset_id="AAPL",
            order_type=OrderType.STOP_LIMIT,
            side=OrderSide.SELL,
            quantity=100,
            stop_price=95.0,
            limit_price=94.5,
        )

        assert order.order_type == OrderType.STOP_LIMIT
        assert order.stop_price == 95.0
        assert order.limit_price == 94.5

    def test_stop_limit_order_missing_prices(self):
        """Test stop-limit order without both prices fails."""
        with pytest.raises(
            ValueError,
            match="Stop-limit orders must have both stop and limit prices",
        ):
            Order(
                asset_id="AAPL",
                order_type=OrderType.STOP_LIMIT,
                side=OrderSide.SELL,
                quantity=100,
                stop_price=95.0,
                # Missing limit_price
            )


class TestTrailingStops:
    """Test suite for trailing stop orders."""

    @pytest.fixture
    def broker(self):
        """Create a broker instance."""
        # Use immediate execution for advanced order tests
        return SimulationBroker(initial_cash=100000.0, execution_delay=False)

    def test_trailing_stop_creation_with_amount(self):
        """Test trailing stop with absolute amount."""
        order = Order(
            asset_id="AAPL",
            order_type=OrderType.TRAILING_STOP,
            side=OrderSide.SELL,
            quantity=100,
            trail_amount=2.0,
        )

        assert order.order_type == OrderType.TRAILING_STOP
        assert order.trail_amount == 2.0
        assert order.trailing_stop_price is None  # Not set until first update

    def test_trailing_stop_creation_with_percent(self):
        """Test trailing stop with percentage."""
        order = Order(
            asset_id="AAPL",
            order_type=OrderType.TRAILING_STOP,
            side=OrderSide.SELL,
            quantity=100,
            trail_percent=2.0,  # 2%
        )

        assert order.order_type == OrderType.TRAILING_STOP
        assert order.trail_percent == 2.0

    def test_trailing_stop_missing_parameters(self):
        """Test trailing stop without trail parameters fails."""
        with pytest.raises(
            ValueError,
            match="Trailing stop orders must have trail_amount or trail_percent",
        ):
            Order(
                asset_id="AAPL",
                order_type=OrderType.TRAILING_STOP,
                side=OrderSide.SELL,
                quantity=100,
            )

    def test_trailing_stop_update_absolute(self):
        """Test trailing stop updating with absolute amount."""
        order = Order(
            asset_id="AAPL",
            order_type=OrderType.TRAILING_STOP,
            side=OrderSide.SELL,
            quantity=100,
            trail_amount=2.0,
        )

        # Initialize at price 100
        assert order.update_trailing_stop(100.0) is True
        assert order.trailing_stop_price == 98.0  # 100 - 2

        # Price rises to 102 - should update trail
        assert order.update_trailing_stop(102.0) is True
        assert order.trailing_stop_price == 100.0  # 102 - 2

        # Price falls to 101 - should not update
        assert order.update_trailing_stop(101.0) is False
        assert order.trailing_stop_price == 100.0  # Unchanged

    def test_trailing_stop_update_percentage(self):
        """Test trailing stop updating with percentage."""
        order = Order(
            asset_id="AAPL",
            order_type=OrderType.TRAILING_STOP,
            side=OrderSide.SELL,
            quantity=100,
            trail_percent=2.0,  # 2%
        )

        # Initialize at price 100
        assert order.update_trailing_stop(100.0) is True
        assert order.trailing_stop_price == 98.0  # 100 - 2%

        # Price rises to 105 - should update trail
        assert order.update_trailing_stop(105.0) is True
        assert order.trailing_stop_price == pytest.approx(102.9, rel=1e-3)  # 105 - 2%

    def test_trailing_stop_triggering(self, broker):
        """Test trailing stop triggering."""
        # First establish a long position
        buy_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )
        broker.submit_order(buy_order)

        # Fill the buy order
        buy_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 9, 30),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.0,
        )
        broker.on_market_event(buy_event)

        # Now submit trailing stop (use percentage for VectorBT-style TSL)
        order = Order(
            asset_id="AAPL",
            order_type=OrderType.TRAILING_STOP,
            side=OrderSide.SELL,
            quantity=100,
            trail_percent=2.0,  # 2% trail
            metadata={"base_price": 100.0},
        )

        broker.submit_order(order)

        # Price rises to 102, trail should update
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 1),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            open=102.0,
            high=102.5,
            low=101.5,
            close=102.0,
        )
        fills = broker.on_market_event(market_event)
        assert len(fills) == 0  # Still should not trigger
        # After this bar: peak = 102.5, TSL = 102.5 * 0.98 = 100.45

        # Price falls to 99.5, should trigger (below trail of ~100.45)
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 2),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            open=100.0,
            high=100.5,
            low=99.5,
            close=99.5,
        )
        fills = broker.on_market_event(market_event)
        assert len(fills) == 1
        assert fills[0].asset_id == "AAPL"


class TestBracketOrders:
    """Test suite for bracket orders."""

    @pytest.fixture
    def broker(self):
        """Create a broker instance."""
        # Use immediate execution for advanced order tests
        return SimulationBroker(initial_cash=100000.0, execution_delay=False)

    def test_bracket_order_creation(self):
        """Test bracket order validation."""
        order = Order(
            asset_id="AAPL",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=100,
            limit_price=99.0,  # Entry price
            profit_target=105.0,  # Take profit
            stop_loss=95.0,  # Stop loss
        )

        assert order.order_type == OrderType.BRACKET
        assert order.profit_target == 105.0
        assert order.stop_loss == 95.0

    def test_bracket_order_missing_parameters(self):
        """Test bracket order without any exit parameters fails."""
        with pytest.raises(
            ValueError,
            match="Bracket orders must have exit parameters",
        ):
            Order(
                asset_id="AAPL",
                order_type=OrderType.BRACKET,
                side=OrderSide.BUY,
                quantity=100,
                limit_price=99.0,
                # Missing ALL exit parameters (profit_target, stop_loss, tp_pct, sl_pct, tsl_pct)
            )

    def test_bracket_order_execution(self, broker):
        """Test bracket order execution and leg creation."""
        # Create bracket order
        bracket_order = Order(
            asset_id="AAPL",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=100,
            limit_price=99.0,
            profit_target=105.0,
            stop_loss=95.0,
        )

        order_id = broker.submit_order(bracket_order)

        # Market price drops to entry level, should fill bracket order
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=99.0,
        )
        fills = broker.on_market_event(market_event)

        # Should have one fill (the initial bracket order)
        assert len(fills) == 1
        assert fills[0].asset_id == "AAPL"
        assert fills[0].fill_quantity == 100

        # Check that bracket legs were created
        filled_order = broker._orders[order_id]
        assert len(filled_order.child_order_ids) == 2  # Stop loss + Take profit

        # Verify child orders exist and are properly configured
        child_orders = [broker._orders[child_id] for child_id in filled_order.child_order_ids]

        # Should have one stop and one limit order
        stop_orders = [o for o in child_orders if o.order_type == OrderType.STOP]
        limit_orders = [o for o in child_orders if o.order_type == OrderType.LIMIT]

        assert len(stop_orders) == 1
        assert len(limit_orders) == 1

        # Both should be sell orders (opposite of original buy)
        assert all(o.side == OrderSide.SELL for o in child_orders)
        assert all(o.quantity == 100 for o in child_orders)

        # Check prices
        stop_order = stop_orders[0]
        limit_order = limit_orders[0]

        assert stop_order.stop_price == 95.0  # Stop loss
        assert limit_order.limit_price == 105.0  # Take profit

    def test_bracket_oco_behavior(self, broker):
        """Test that bracket legs cancel each other (OCO behavior)."""
        # Create and fill bracket order
        bracket_order = Order(
            asset_id="AAPL",
            order_type=OrderType.BRACKET,
            side=OrderSide.BUY,
            quantity=100,
            limit_price=99.0,
            profit_target=105.0,
            stop_loss=95.0,
        )

        order_id = broker.submit_order(bracket_order)

        # Fill the bracket order
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=99.0,
        )
        fills = broker.on_market_event(market_event)
        assert len(fills) == 1

        # Get child order IDs
        filled_order = broker._orders[order_id]
        child_ids = filled_order.child_order_ids
        assert len(child_ids) == 2

        # Price rises to take profit level
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 1),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=105.0,
        )
        fills = broker.on_market_event(market_event)

        # Should fill the take profit order
        assert len(fills) == 1

        # The other leg should be cancelled
        child_orders = [broker._orders[child_id] for child_id in child_ids]
        filled_orders = [o for o in child_orders if o.is_filled]
        cancelled_orders = [o for o in child_orders if o.state == OrderState.CANCELLED]

        assert len(filled_orders) == 1
        assert len(cancelled_orders) == 1


class TestOrderIntegration:
    """Integration tests for advanced order types."""

    @pytest.fixture
    def broker(self):
        """Create a broker instance."""
        # Use immediate execution for advanced order tests
        return SimulationBroker(initial_cash=100000.0, execution_delay=False)

    def test_mixed_order_types_execution(self, broker):
        """Test that different order types can coexist and execute properly."""
        # Submit various order types
        market_order = Order(
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=50,
        )

        limit_order = Order(
            asset_id="AAPL",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=100,
            limit_price=98.0,
        )

        stop_order = Order(
            asset_id="AAPL",
            order_type=OrderType.STOP,
            side=OrderSide.SELL,
            quantity=75,
            stop_price=97.0,
        )

        trailing_stop = Order(
            asset_id="AAPL",
            order_type=OrderType.TRAILING_STOP,
            side=OrderSide.SELL,
            quantity=25,
            trail_percent=1.0,  # Use percentage for TSL
            metadata={"base_price": 100.0},  # Initialize peak
        )

        # Submit all orders
        market_id = broker.submit_order(market_order)
        limit_id = broker.submit_order(limit_order)
        stop_id = broker.submit_order(stop_order)
        trail_id = broker.submit_order(trailing_stop)

        # Market event at 100
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.0,
        )
        fills = broker.on_market_event(market_event)

        # Should fill market order only
        assert len(fills) == 1
        assert fills[0].order_id == market_id

        # Market event at 98 - should fill limit order
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 1),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            open=99.0,
            high=99.5,
            low=98.0,
            close=98.0,
        )
        fills = broker.on_market_event(market_event)

        # Should fill limit order AND trailing stop (which triggers at 99.0)
        assert len(fills) == 2
        filled_ids = {fill.order_id for fill in fills}
        assert limit_id in filled_ids  # Limit order at 98.0
        assert trail_id in filled_ids  # Trailing stop triggers below 99.0

        # Market event at 97 - should trigger stop order
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 2),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            open=98.0,
            high=98.5,
            low=97.0,
            close=97.0,
        )
        fills = broker.on_market_event(market_event)

        # Should trigger stop order only (trailing stop already filled)
        assert len(fills) == 1
        assert fills[0].order_id == stop_id
