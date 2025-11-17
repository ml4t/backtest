"""Integration tests for RiskManager with position state tracking.

Tests the full workflow: record_fill -> check_position_exits -> bars_held tracking.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from ml4t.backtest.risk.manager import RiskManager, PositionTradeState
from ml4t.backtest.risk.rules.time_based import TimeBasedExit
from ml4t.backtest.core.event import MarketEvent, FillEvent
from ml4t.backtest.core.types import MarketDataType, OrderSide
from ml4t.backtest.portfolio.state import Position
from ml4t.backtest.portfolio.portfolio import Portfolio


def make_fill_event(
    asset_id: str = "AAPL",
    timestamp: datetime = None,
    side: OrderSide = OrderSide.BUY,
    fill_quantity: float = 10.0,
    fill_price: Decimal = Decimal("100.0"),
    commission: float = 1.0,
    order_id: str = "order-001",
    trade_id: str = "trade-001",
) -> FillEvent:
    """Helper to create FillEvent with sensible defaults."""
    if timestamp is None:
        timestamp = datetime(2024, 1, 1, 10, 0)

    return FillEvent(
        timestamp=timestamp,
        order_id=order_id,
        trade_id=trade_id,
        asset_id=asset_id,
        side=side,
        fill_quantity=fill_quantity,
        fill_price=fill_price,
        commission=commission,
    )


def make_market_event(
    asset_id: str = "AAPL",
    timestamp: datetime = None,
    close: Decimal = Decimal("100.0"),
    open: Decimal = None,
    high: Decimal = None,
    low: Decimal = None,
    volume: int = 1000,
) -> MarketEvent:
    """Helper to create MarketEvent with sensible defaults."""
    if timestamp is None:
        timestamp = datetime(2024, 1, 1, 10, 0)
    if open is None:
        open = close - Decimal("1.0")
    if high is None:
        high = close + Decimal("1.0")
    if low is None:
        low = close - Decimal("2.0")

    return MarketEvent(
        timestamp=timestamp,
        asset_id=asset_id,
        data_type=MarketDataType.BAR,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


class MockBroker:
    """Mock broker for testing."""

    def __init__(self):
        self.positions = {}

    def get_position(self, asset_id):
        """Return Position object (not just quantity)."""
        return self.positions.get(asset_id)

    def get_positions(self):
        """Return dict of Position objects."""
        return {k: v for k, v in self.positions.items() if v.quantity != 0}

    def set_position(self, asset_id, quantity, cost_basis):
        """Set a position for testing."""
        self.positions[asset_id] = Position(
            asset_id=asset_id,
            quantity=quantity,
            cost_basis=cost_basis,
            last_price=cost_basis / quantity if quantity != 0 else 0.0,
        )


class TestRiskManagerPositionTracking:
    """Test RiskManager position state tracking."""

    def test_record_fill_creates_position_state(self):
        """Test that record_fill creates PositionTradeState on entry."""
        manager = RiskManager()

        # Create fill event (buy 10 shares at $100)
        entry_time = datetime(2024, 1, 1, 10, 0)
        fill_event = make_fill_event(timestamp=entry_time)
        market_event = make_market_event(timestamp=entry_time)

        # Record fill
        manager.record_fill(fill_event, market_event)

        # Verify PositionTradeState was created
        assert "AAPL" in manager._position_state
        state = manager._position_state["AAPL"]

        assert state.asset_id == "AAPL"
        assert state.entry_time == entry_time
        assert state.entry_price == Decimal("100.00")
        assert state.entry_quantity == 10.0
        assert state.bars_held == 0
        assert state.max_favorable_excursion == Decimal("0.0")
        assert state.max_adverse_excursion == Decimal("0.0")

    def test_check_position_exits_updates_bars_held(self):
        """Test that check_position_exits increments bars_held."""
        manager = RiskManager()
        broker = MockBroker()
        portfolio = Portfolio(initial_cash=100_000)

        # Set up position
        entry_time = datetime(2024, 1, 1, 10, 0)
        broker.set_position("AAPL", quantity=10.0, cost_basis=1000.0)

        # Create position state manually
        manager._position_state["AAPL"] = PositionTradeState(
            asset_id="AAPL",
            entry_time=entry_time,
            entry_price=Decimal("100.00"),
            entry_quantity=10.0,
        )

        # Simulate 3 market events
        for i in range(3):
            event_time = entry_time + timedelta(minutes=i+1)
            market_event = MarketEvent(
                timestamp=event_time,
                asset_id="AAPL",
                close=Decimal("101.00"),
            )

            # Check position exits (this should update bars_held)
            exit_orders = manager.check_position_exits(
                market_event=market_event,
                broker=broker,
                portfolio=portfolio,
            )

        # Verify bars_held was incremented
        state = manager._position_state["AAPL"]
        assert state.bars_held == 3

    def test_check_position_exits_updates_mfe_mae(self):
        """Test that check_position_exits updates MFE/MAE."""
        manager = RiskManager()
        broker = MockBroker()
        portfolio = Portfolio(initial_cash=100_000)

        # Set up long position at $100
        entry_time = datetime(2024, 1, 1, 10, 0)
        broker.set_position("AAPL", quantity=10.0, cost_basis=1000.0)

        manager._position_state["AAPL"] = PositionTradeState(
            asset_id="AAPL",
            entry_time=entry_time,
            entry_price=Decimal("100.00"),
            entry_quantity=10.0,
        )

        # Bar 1: Price goes up to 105 (favorable)
        event1 = MarketEvent(
            timestamp=entry_time + timedelta(minutes=1),
            asset_id="AAPL",
            close=Decimal("105.00"),
        )
        manager.check_position_exits(event1, broker, portfolio)

        state = manager._position_state["AAPL"]
        assert state.max_favorable_excursion == Decimal("5.00")  # 105 - 100
        assert state.max_adverse_excursion == Decimal("0.0")

        # Bar 2: Price drops to 95 (adverse)
        event2 = MarketEvent(
            timestamp=entry_time + timedelta(minutes=2),
            asset_id="AAPL",
            close=Decimal("95.00"),
        )
        manager.check_position_exits(event2, broker, portfolio)

        state = manager._position_state["AAPL"]
        assert state.max_favorable_excursion == Decimal("5.00")  # Still at peak
        assert state.max_adverse_excursion == Decimal("5.00")  # 100 - 95

        # Bar 3: Price at 102 (between peak and entry)
        event3 = MarketEvent(
            timestamp=entry_time + timedelta(minutes=3),
            asset_id="AAPL",
            close=Decimal("102.00"),
        )
        manager.check_position_exits(event3, broker, portfolio)

        state = manager._position_state["AAPL"]
        assert state.max_favorable_excursion == Decimal("5.00")  # Still at peak
        assert state.max_adverse_excursion == Decimal("5.00")  # Still at worst

    def test_time_based_exit_triggers_at_exact_bar_count(self):
        """Test that TimeBasedExit triggers at exactly max_bars."""
        manager = RiskManager()
        manager.add_rule(TimeBasedExit(max_bars=5))

        broker = MockBroker()
        portfolio = Portfolio(initial_cash=100_000)

        # Set up position
        entry_time = datetime(2024, 1, 1, 10, 0)
        broker.set_position("AAPL", quantity=10.0, cost_basis=1000.0)

        manager._position_state["AAPL"] = PositionTradeState(
            asset_id="AAPL",
            entry_time=entry_time,
            entry_price=Decimal("100.00"),
            entry_quantity=10.0,
        )

        # Bars 1-4: Should not trigger
        for i in range(1, 5):
            event_time = entry_time + timedelta(minutes=i)
            market_event = MarketEvent(
                timestamp=event_time,
                asset_id="AAPL",
                close=Decimal("101.00"),
            )

            exit_orders = manager.check_position_exits(
                market_event=market_event,
                broker=broker,
                portfolio=portfolio,
            )

            assert len(exit_orders) == 0, f"Should not exit at bar {i}"
            assert manager._position_state["AAPL"].bars_held == i

        # Bar 5: Should trigger exit
        event_time = entry_time + timedelta(minutes=5)
        market_event = MarketEvent(
            timestamp=event_time,
            asset_id="AAPL",
            close=Decimal("101.00"),
        )

        exit_orders = manager.check_position_exits(
            market_event=market_event,
            broker=broker,
            portfolio=portfolio,
        )

        assert len(exit_orders) == 1, "Should generate exit order at bar 5"
        assert exit_orders[0].asset_id == "AAPL"
        assert exit_orders[0].quantity == 10.0  # Exit full position
        assert manager._position_state["AAPL"].bars_held == 5

    def test_position_state_removed_on_close(self):
        """Test that position state is removed when position is closed."""
        manager = RiskManager()

        # Create entry fill
        entry_time = datetime(2024, 1, 1, 10, 0)
        entry_fill = make_fill_event(timestamp=entry_time, order_id="entry")
        entry_event = MarketEvent(
            timestamp=entry_time,
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            open=Decimal("99.0"),
            high=Decimal("101.0"),
            low=Decimal("98.0"),
            close=Decimal("100.00"),
            volume=1000,
        )

        manager.record_fill(entry_fill, entry_event)
        assert "AAPL" in manager._position_state

        # Create exit fill (closes position)
        exit_time = entry_time + timedelta(minutes=5)
        exit_fill = make_fill_event(
            timestamp=exit_time,
            side=OrderSide.SELL,
            fill_price=Decimal("105.00"),
            order_id="exit",
            trade_id="exit-trade",
        )
        exit_event = MarketEvent(
            timestamp=exit_time,
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            open=Decimal("104.0"),
            high=Decimal("106.0"),
            low=Decimal("103.0"),
            close=Decimal("105.00"),
            volume=1000,
        )

        manager.record_fill(exit_fill, exit_event)

        # Position state should be removed
        assert "AAPL" not in manager._position_state
        assert "AAPL" not in manager._position_levels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
