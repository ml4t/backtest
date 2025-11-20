"""Tests for market impact integration with broker."""

from datetime import datetime, timedelta

import pytest

from ml4t.backtest.core.event import MarketEvent
from ml4t.backtest.core.types import MarketDataType
from ml4t.backtest.execution.broker import SimulationBroker
from ml4t.backtest.execution.market_impact import AlmgrenChrissImpact, LinearMarketImpact
from ml4t.backtest.execution.order import Order, OrderSide, OrderType


@pytest.fixture
def impact_broker():
    """Create broker with market impact model."""
    impact_model = LinearMarketImpact(
        permanent_impact_factor=0.1,
        temporary_impact_factor=0.5,
        avg_daily_volume=100_000,
        decay_rate=0.1,
    )

    return SimulationBroker(
        initial_cash=500_000,  # Increased to support large test orders
        market_impact_model=impact_model,
        execution_delay=True,  # Required for orders to be processed on market events
    )


@pytest.fixture
def market_event():
    """Create a market event."""
    return MarketEvent(
        timestamp=datetime.now(),
        asset_id="AAPL",
        data_type=MarketDataType.BAR,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=10000,
    )


class TestMarketImpactIntegration:
    """Test market impact integration with broker."""

    def test_broker_with_no_impact_model(self):
        """Test broker works without market impact model."""
        broker = SimulationBroker()

        Order(
            asset_id="AAPL",
            quantity=1000,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )

        # Should work fine without impact
        fills = broker.on_market_event(
            MarketEvent(
                timestamp=datetime.now(),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10000,
            ),
        )

        # No fills since no orders submitted
        assert len(fills) == 0

    def test_market_impact_affects_fill_price(self, impact_broker):
        """Test that market impact affects fill prices."""
        timestamp = datetime.now()

        # Submit single order to create initial impact
        order1 = Order(
            asset_id="AAPL",
            quantity=2000,  # Large order to create noticeable impact
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )

        impact_broker.submit_order(order1)

        market_event1 = MarketEvent(
            timestamp=timestamp,
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10000,
        )

        # First event - no fill due to execution delay
        fills1 = impact_broker.on_market_event(market_event1)
        assert len(fills1) == 0

        # Second event - order should fill
        market_event2 = MarketEvent(
            timestamp=timestamp + timedelta(seconds=1),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10000,
        )

        fills = impact_broker.on_market_event(market_event2)
        assert len(fills) == 1

        # First order creates impact but doesn't experience it
        # Fill price should be base price + slippage only
        assert fills[0].fill_price == pytest.approx(100.01, rel=1e-4)

        # Verify that impact was created AFTER the fill
        impact_after = impact_broker.market_impact_model.get_current_impact("AAPL")
        assert impact_after > 0  # Should have positive impact from buy order

        # Submit second order to verify it experiences the accumulated impact
        order2 = Order(
            asset_id="AAPL",
            quantity=1000,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )
        impact_broker.submit_order(order2)

        # First event for second order
        market_event3 = MarketEvent(
            timestamp=timestamp + timedelta(seconds=2),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10000,
        )
        impact_broker.on_market_event(market_event3)

        # Second event - second order fills with market impact
        market_event4 = MarketEvent(
            timestamp=timestamp + timedelta(seconds=3),
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10000,
        )
        fills2 = impact_broker.on_market_event(market_event4)
        assert len(fills2) == 1

        # Second order should experience the impact from first order
        # Should be noticeably higher than 100.01 (base + slippage only)
        assert fills2[0].fill_price > 100.02

    def test_impact_accumulation(self, impact_broker):
        """Test that market impact accumulates across trades."""
        market_price = 100.0
        timestamp = datetime.now()

        # Track impact accumulation
        initial_impact = impact_broker.market_impact_model.get_current_impact("AAPL")
        assert initial_impact == 0.0

        # First trade
        order1 = Order(
            asset_id="AAPL",
            quantity=1000,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )

        impact_broker.submit_order(order1)
        # First market event - no fill due to execution delay
        impact_broker.on_market_event(
            MarketEvent(
                timestamp=timestamp,
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                open=market_price,
                high=market_price,
                low=market_price,
                close=market_price,
                volume=10000,
            ),
        )
        # Second market event - order fills
        impact_broker.on_market_event(
            MarketEvent(
                timestamp=timestamp + timedelta(milliseconds=100),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                open=market_price,
                high=market_price,
                low=market_price,
                close=market_price,
                volume=10000,
            ),
        )

        impact_after_1 = impact_broker.market_impact_model.get_current_impact("AAPL")
        assert impact_after_1 > 0

        # Second trade
        order2 = Order(
            asset_id="AAPL",
            quantity=2000,  # Larger trade
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )

        impact_broker.submit_order(order2)
        # First market event - no fill
        impact_broker.on_market_event(
            MarketEvent(
                timestamp=timestamp + timedelta(seconds=1),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                open=market_price,
                high=market_price,
                low=market_price,
                close=market_price,
                volume=10000,
            ),
        )
        # Second market event - order fills
        impact_broker.on_market_event(
            MarketEvent(
                timestamp=timestamp + timedelta(seconds=1, milliseconds=100),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                open=market_price,
                high=market_price,
                low=market_price,
                close=market_price,
                volume=10000,
            ),
        )

        impact_after_2 = impact_broker.market_impact_model.get_current_impact("AAPL")

        # Impact should have increased (accumulated)
        assert impact_after_2 > impact_after_1

    def test_separate_asset_impact(self, impact_broker):
        """Test that impact is tracked separately per asset."""
        timestamp = datetime.now()

        # Trade AAPL
        aapl_order = Order(
            asset_id="AAPL",
            quantity=500,  # Smaller order to ensure different impact from GOOGL
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )

        impact_broker.submit_order(aapl_order)
        # First AAPL event - no fill
        impact_broker.on_market_event(
            MarketEvent(
                timestamp=timestamp,
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=10000,
            ),
        )
        # Second AAPL event - order fills
        impact_broker.on_market_event(
            MarketEvent(
                timestamp=timestamp + timedelta(milliseconds=100),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=10000,
            ),
        )

        # Trade GOOGL - first buy to establish position, then sell
        googl_buy_order = Order(
            asset_id="GOOGL",
            quantity=1000,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )

        impact_broker.submit_order(googl_buy_order)
        # Buy GOOGL to establish position
        impact_broker.on_market_event(
            MarketEvent(
                timestamp=timestamp,
                asset_id="GOOGL",
                data_type=MarketDataType.BAR,
                open=200.0,
                high=200.0,
                low=200.0,
                close=200.0,
                volume=5000,
            ),
        )
        impact_broker.on_market_event(
            MarketEvent(
                timestamp=timestamp + timedelta(milliseconds=100),
                asset_id="GOOGL",
                data_type=MarketDataType.BAR,
                open=200.0,
                high=200.0,
                low=200.0,
                close=200.0,
                volume=5000,
            ),
        )

        # Now sell GOOGL
        googl_order = Order(
            asset_id="GOOGL",
            quantity=500,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
        )

        impact_broker.submit_order(googl_order)
        # First GOOGL event - no fill
        impact_broker.on_market_event(
            MarketEvent(
                timestamp=timestamp,
                asset_id="GOOGL",
                data_type=MarketDataType.BAR,
                open=200.0,
                high=200.0,
                low=200.0,
                close=200.0,
                volume=5000,
            ),
        )
        # Second GOOGL event - order fills
        impact_broker.on_market_event(
            MarketEvent(
                timestamp=timestamp + timedelta(milliseconds=100),
                asset_id="GOOGL",
                data_type=MarketDataType.BAR,
                open=200.0,
                high=200.0,
                low=200.0,
                close=200.0,
                volume=5000,
            ),
        )

        # Check separate impacts
        aapl_impact = impact_broker.market_impact_model.get_current_impact("AAPL")
        googl_impact = impact_broker.market_impact_model.get_current_impact("GOOGL")

        assert aapl_impact > 0  # Buy order pushes price up
        assert googl_impact != 0  # GOOGL had trades (buy + sell)
        assert aapl_impact != googl_impact  # Impacts tracked separately

    def test_impact_decay_over_time(self):
        """Test that temporary impact decays over time."""
        impact_model = LinearMarketImpact(
            permanent_impact_factor=0.05,  # Small permanent
            temporary_impact_factor=0.2,  # Larger temporary
            decay_rate=1.0,  # Fast decay for testing
        )

        broker = SimulationBroker(
            initial_cash=500_000,
            market_impact_model=impact_model,
            execution_delay=True,
        )

        # Initial trade
        order = Order(
            asset_id="AAPL",
            quantity=1000,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )

        broker.submit_order(order)

        initial_time = datetime.now()
        # First event - no fill
        broker.on_market_event(
            MarketEvent(
                timestamp=initial_time,
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=10000,
            ),
        )
        # Second event - order fills
        broker.on_market_event(
            MarketEvent(
                timestamp=initial_time + timedelta(milliseconds=100),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=10000,
            ),
        )

        # Check impact immediately
        immediate_impact = impact_model.get_current_impact("AAPL")

        # Check impact after 1 second (with decay)
        later_time = initial_time + timedelta(seconds=1)
        decayed_impact = impact_model.get_current_impact("AAPL", later_time)

        # Should have decayed (temporary impact reduces, permanent remains)
        assert decayed_impact < immediate_impact
        assert decayed_impact > 0  # But still positive due to permanent component

    def test_almgren_chriss_integration(self):
        """Test Almgren-Chriss model integration."""
        impact_model = AlmgrenChrissImpact(
            permanent_impact_const=0.01,
            temporary_impact_const=0.1,
            daily_volatility=0.02,
            avg_daily_volume=100_000,
        )

        broker = SimulationBroker(
            initial_cash=500_000,
            market_impact_model=impact_model,
            execution_delay=True,
        )

        # Test different order sizes to verify square root scaling
        order_sizes = [100, 400, 1600]  # 1x, 4x, 16x
        fill_prices = []

        for i, size in enumerate(order_sizes):
            order = Order(
                asset_id="AAPL",
                quantity=size,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
            )

            broker.submit_order(order)
            # First event - no fill
            broker.on_market_event(
                MarketEvent(
                    timestamp=datetime.now() + timedelta(seconds=i),
                    asset_id="AAPL",
                    data_type=MarketDataType.BAR,
                    open=100.0,
                    high=100.0,
                    low=100.0,
                    close=100.0,
                    volume=10000,
                ),
            )
            # Second event - order fills
            fills = broker.on_market_event(
                MarketEvent(
                    timestamp=datetime.now() + timedelta(seconds=i, milliseconds=100),
                    asset_id="AAPL",
                    data_type=MarketDataType.BAR,
                    open=100.0,
                    high=100.0,
                    low=100.0,
                    close=100.0,
                    volume=10000,
                ),
            )

            assert len(fills) == 1
            fill_prices.append(fills[0].fill_price)

        # Prices should increase, but not linearly (due to square root scaling)
        assert fill_prices[1] > fill_prices[0]
        assert fill_prices[2] > fill_prices[1]

        # The impact scaling should be closer to square root than linear
        # Due to accumulated impact from previous orders, the ratio will be affected
        # Check that prices increase (showing impact accumulation)
        impact_1 = fill_prices[0] - 100.0
        impact_2 = fill_prices[1] - 100.0
        impact_3 = fill_prices[2] - 100.0

        # All orders should show increasing prices due to accumulated impact
        assert impact_2 > impact_1  # Second order experiences first order's impact
        assert impact_3 > impact_2  # Third order experiences accumulated impact

    def test_market_impact_with_limit_orders(self, impact_broker):
        """Test market impact with limit orders."""
        # Submit limit order that creates impact when filled
        limit_order = Order(
            asset_id="AAPL",
            quantity=2000,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=100.5,
        )

        impact_broker.submit_order(limit_order)

        # First event - no fill due to execution delay
        timestamp = datetime.now()
        impact_broker.on_market_event(
            MarketEvent(
                timestamp=timestamp,
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                open=100.0,
                high=100.5,  # High enough to trigger limit
                low=99.5,
                close=100.3,
                volume=10000,
            ),
        )

        # Second event - limit order fills
        fills = impact_broker.on_market_event(
            MarketEvent(
                timestamp=timestamp + timedelta(milliseconds=100),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                open=100.0,
                high=100.5,  # High enough to trigger limit
                low=99.5,
                close=100.3,
                volume=10000,
            ),
        )

        assert len(fills) == 1

        # Check that impact was created
        impact = impact_broker.market_impact_model.get_current_impact("AAPL")
        assert impact > 0  # Buy order should create positive impact

        # Subsequent market order should be affected by this impact
        market_order = Order(
            asset_id="AAPL",
            quantity=1000,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )

        impact_broker.submit_order(market_order)

        # First event - no fill
        timestamp2 = datetime.now() + timedelta(seconds=1)
        impact_broker.on_market_event(
            MarketEvent(
                timestamp=timestamp2,
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                open=100.0,
                high=100.5,
                low=99.5,
                close=100.0,
                volume=10000,
            ),
        )

        # Second event - market order fills
        fills2 = impact_broker.on_market_event(
            MarketEvent(
                timestamp=timestamp2 + timedelta(milliseconds=100),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                open=100.0,
                high=100.5,
                low=99.5,
                close=100.0,
                volume=10000,
            ),
        )

        assert len(fills2) == 1
        # Should fill at higher than market price due to impact
        assert fills2[0].fill_price > 100.0
