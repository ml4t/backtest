"""Test cash constraint handling in broker to prevent negative fill quantities."""

from datetime import datetime, timezone

import pytest

from qengine.core.event import MarketEvent
from qengine.core.types import MarketDataType, OrderSide, OrderType
from qengine.execution.broker import SimulationBroker
from qengine.execution.commission import FlatCommission, PercentageCommission
from qengine.execution.order import Order
from qengine.execution.slippage import NoSlippage


class TestCashConstraints:
    """Test suite for cash-constrained order fills."""

    def test_no_negative_fill_quantity_with_commission(self):
        """Test that fill quantity is never negative even with high commission."""
        # Create broker with limited cash and high commission
        broker = SimulationBroker(
            initial_cash=1000.0,
            commission_model=FlatCommission(commission=10.0),  # $10 flat fee
            execution_delay=False,  # Immediate execution for testing
        )

        timestamp = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)

        # Try to buy stock at $100 with only $1000 cash
        # Without commission: could buy 10 shares
        # With $10 commission: can buy 9 shares ($900 + $10 = $910 < $1000)
        order = Order(
            order_id="TEST001",
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100.0,  # Want to buy 100 shares
        )

        # Submit order
        broker.submit_order(order, timestamp)

        # Create market event to trigger fill
        market_event = MarketEvent(
            timestamp=timestamp,
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            price=100.0,
            close=100.0,
            volume=10000,
        )

        # Process market event
        fills = broker.on_market_event(market_event)

        # Should get a partial fill within cash constraints
        assert len(fills) == 1
        fill = fills[0]

        # Verify fill quantity is positive and within cash limits
        assert fill.fill_quantity > 0, "Fill quantity must be positive"
        assert fill.fill_quantity < 10.0, "Should be less than full quantity due to commission"

        # Verify total cost doesn't exceed available cash
        total_cost = fill.fill_quantity * fill.fill_price + fill.commission
        assert total_cost <= 1000.0, "Total cost should not exceed available cash"

        # Verify we got close to the expected ~9 shares (allowing for slippage)
        assert 8.5 <= fill.fill_quantity <= 9.5, "Should fill approximately 9 shares"

    def test_percentage_commission_cash_constraint(self):
        """Test cash constraints with percentage-based commission."""
        broker = SimulationBroker(
            initial_cash=1000.0,
            commission_model=PercentageCommission(rate=0.01),  # 1% commission
            execution_delay=False,
        )

        timestamp = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)

        # Try to buy at $50 per share
        # Without commission: could buy 20 shares
        # With 1% commission: can buy ~19.8 shares
        # (19.8 * 50 * 1.01 = 999.9 < 1000)
        order = Order(
            order_id="TEST002",
            asset_id="MSFT",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=50.0,  # Want to buy 50 shares
        )

        broker.submit_order(order, timestamp)

        market_event = MarketEvent(
            timestamp=timestamp,
            asset_id="MSFT",
            data_type=MarketDataType.BAR,
            price=50.0,
            close=50.0,
            volume=10000,
        )

        fills = broker.on_market_event(market_event)

        assert len(fills) == 1
        fill = fills[0]

        # Check fill is positive and within constraints
        assert fill.fill_quantity > 0
        assert fill.fill_quantity < 20.0  # Less than what we could buy without commission

        # Verify total cost (use approx due to floating point precision)
        total_cost = fill.fill_quantity * fill.fill_price * 1.01  # Including 1% commission
        assert total_cost <= 1000.0 + 0.01  # Allow tiny floating point error

    def test_insufficient_cash_for_commission(self):
        """Test that order is rejected if can't even afford commission."""
        broker = SimulationBroker(
            initial_cash=5.0,  # Only $5 available
            commission_model=FlatCommission(commission=10.0),  # $10 flat fee
            execution_delay=False,
        )

        timestamp = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)

        order = Order(
            order_id="TEST003",
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1.0,
        )

        broker.submit_order(order, timestamp)

        market_event = MarketEvent(
            timestamp=timestamp,
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            price=100.0,
            close=100.0,
            volume=10000,
        )

        fills = broker.on_market_event(market_event)

        # Order should not fill as we can't afford commission
        assert len(fills) == 0

    def test_minimal_cash_exact_fill(self):
        """Test edge case where cash exactly matches required amount."""
        broker = SimulationBroker(
            initial_cash=101.0,  # Exactly enough for 1 share at $100 + $1 commission
            commission_model=FlatCommission(commission=1.0),
            slippage_model=NoSlippage(),
            execution_delay=False,
        )

        timestamp = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)

        order = Order(
            order_id="TEST004",
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1.0,
        )

        broker.submit_order(order, timestamp)

        market_event = MarketEvent(
            timestamp=timestamp,
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            price=100.0,
            close=100.0,
            volume=10000,
        )

        fills = broker.on_market_event(market_event)

        assert len(fills) == 1
        fill = fills[0]
        assert fill.fill_quantity == 1.0
        assert fill.fill_price == 100.0
        assert fill.commission == 1.0

        # Verify broker cash is now ~zero
        assert abs(broker.cash) < 0.01

    def test_no_commission_model(self):
        """Test cash constraints work correctly with no commission model."""
        broker = SimulationBroker(
            initial_cash=1000.0,
            commission_model=None,  # No commission
            slippage_model=NoSlippage(),
            execution_delay=False,
        )

        timestamp = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)

        # Can buy exactly 10 shares at $100 with $1000
        order = Order(
            order_id="TEST005",
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=20.0,  # Want more than we can afford
        )

        broker.submit_order(order, timestamp)

        market_event = MarketEvent(
            timestamp=timestamp,
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            price=100.0,
            close=100.0,
            volume=10000,
        )

        fills = broker.on_market_event(market_event)

        assert len(fills) == 1
        fill = fills[0]

        # Should get approximately 10 shares (slightly less due to commission fallback)
        assert fill.fill_quantity <= 10.0
        assert fill.fill_quantity > 9.95  # Allow for default commission

    def test_sell_order_not_affected_by_cash(self):
        """Test that sell orders are not constrained by cash."""
        broker = SimulationBroker(
            initial_cash=0.0,  # No cash
            commission_model=FlatCommission(commission=10.0),
            slippage_model=NoSlippage(),
            execution_delay=False,
        )

        # Give broker some shares to sell (using Portfolio API)
        broker._internal_portfolio.update_position(
            asset_id="AAPL",
            quantity_change=10.0,
            price=100.0,
        )

        timestamp = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)

        order = Order(
            order_id="TEST006",
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=10.0,
        )

        broker.submit_order(order, timestamp)

        market_event = MarketEvent(
            timestamp=timestamp,
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.0,
            volume=10000,
        )

        fills = broker.on_market_event(market_event)

        # Sell should succeed even with no cash (commission deducted from proceeds)
        assert len(fills) == 1
        fill = fills[0]
        assert fill.fill_quantity == 10.0

        # Cash should be sale proceeds minus commission
        expected_cash = 10.0 * 100.0 - 10.0  # $1000 - $10
        assert abs(broker.cash - expected_cash) < 0.01

    def test_progressive_cash_depletion(self):
        """Test multiple orders progressively depleting cash."""
        broker = SimulationBroker(
            initial_cash=1000.0,
            commission_model=PercentageCommission(rate=0.001),  # 0.1% commission
            slippage_model=NoSlippage(),
            execution_delay=False,
        )

        timestamp = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)

        # Buy orders at different prices
        orders_and_prices = [
            ("AAPL", 100.0, 5.0),   # Buy 5 shares at $100
            ("MSFT", 50.0, 8.0),     # Buy 8 shares at $50
            ("GOOGL", 200.0, 10.0),  # Try to buy 10 shares at $200 (won't have enough)
        ]

        total_spent = 0.0

        for asset_id, price, quantity in orders_and_prices:
            order = Order(
                order_id=f"ORDER_{asset_id}",
                asset_id=asset_id,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                quantity=quantity,
            )

            broker.submit_order(order, timestamp)

            market_event = MarketEvent(
                timestamp=timestamp,
                asset_id=asset_id,
                data_type=MarketDataType.BAR,
                price=price,
                close=price,
                volume=10000,
            )

            fills = broker.on_market_event(market_event)

            if fills:
                fill = fills[0]
                total_spent += fill.fill_quantity * fill.fill_price + fill.commission

                # Each fill should respect remaining cash
                assert broker.cash >= -0.01  # Allow small rounding error

        # Total spent should not exceed initial cash
        assert total_spent <= 1000.01  # Allow small rounding error

    def test_minimum_fill_size_constraint(self):
        """Test that orders below minimum fill size are rejected."""
        broker = SimulationBroker(
            initial_cash=0.5,  # Only $0.50 available
            commission_model=None,  # No commission
            execution_delay=False,
        )

        timestamp = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)

        # At $100 per share, we can only afford 0.005 shares (below 0.01 minimum)
        order = Order(
            order_id="TEST007",
            asset_id="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1.0,
        )

        broker.submit_order(order, timestamp)

        market_event = MarketEvent(
            timestamp=timestamp,
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            price=100.0,
            close=100.0,
            volume=10000,
        )

        fills = broker.on_market_event(market_event)

        # Order should not fill as quantity would be below minimum
        assert len(fills) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
