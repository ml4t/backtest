"""Integration tests for basic risk rules with BacktestEngine.

These tests verify that risk rules work correctly when integrated with
the full backtesting engine and real data.
"""

import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import pytest

from ml4t.backtest.data.polars_feed import PolarsDataFeed
from ml4t.backtest.engine import BacktestEngine
from ml4t.backtest.execution.broker import SimulationBroker
from ml4t.backtest.portfolio.portfolio import Portfolio
from ml4t.backtest.risk import (
    RiskManager,
    TimeBasedExit,
    PriceBasedStopLoss,
    PriceBasedTakeProfit,
)
from ml4t.backtest.strategy.base import Strategy
from ml4t.backtest.core.event import MarketEvent


@pytest.fixture
def temp_data_file(tmp_path):
    """Create a temporary Parquet file with test data.

    Args:
        tmp_path: pytest tmp_path fixture

    Returns:
        Path to temporary Parquet file
    """
    data = _create_test_data(days=100)
    file_path = tmp_path / "test_data.parquet"
    data.write_parquet(file_path)
    return file_path


def _create_test_data(days: int = 100) -> pl.DataFrame:
    """Create synthetic price data for testing.

    Args:
        days: Number of days of data to generate

    Returns:
        DataFrame with OHLCV data
    """
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    # Create trending data: starts at 100, trends up to 120, then down to 90
    prices = []
    for i in range(days):
        if i < 40:
            # Uptrend: 100 -> 120
            price = 100 + (i / 40) * 20
        elif i < 80:
            # Downtrend: 120 -> 90
            price = 120 - ((i - 40) / 40) * 30
        else:
            # Sideways: around 90
            price = 90 + ((i % 5) - 2) * 2

        # Add some intraday range
        high = price + 1.0
        low = price - 1.0
        open_price = price - 0.5
        close_price = price + 0.5

        prices.append(
            {
                "timestamp": dates[i],
                "asset_id": "TEST",
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price,
                "volume": 1_000_000,
            }
        )

    return pl.DataFrame(prices)


class SimpleStrategy(Strategy):
    """Simple buy-and-hold strategy for testing."""

    def __init__(self, entry_bar: int = 5):
        """Initialize strategy.

        Args:
            entry_bar: Bar number to enter position
        """
        super().__init__()
        self.entry_bar = entry_bar
        self.bar_count = 0
        self.entered = False

    def on_event(self, event) -> None:
        """Handle all events (required abstract method)."""
        # Delegate to specific handlers
        pass

    def on_market_data(self, event: MarketEvent) -> None:
        """Enter position on specified bar."""
        self.bar_count += 1

        if self.bar_count == self.entry_bar and not self.entered:
            # Buy 100 shares at market
            self.buy_percent(event.asset_id, percent=0.5, limit_price=None)
            self.entered = True


class TestTimeBasedExitIntegration:
    """Integration tests for TimeBasedExit rule."""

    def test_time_based_exit_closes_position(self, temp_data_file):
        """Test that TimeBasedExit closes position after max_bars."""
        # Create data and components
        feed = PolarsDataFeed(temp_data_file, asset_id="TEST")
        broker = SimulationBroker(initial_cash=100_000)

        # Create risk manager with time-based exit (max 10 bars)
        risk_manager = RiskManager()
        risk_manager.add_rule(TimeBasedExit(max_bars=10))

        # Create strategy that enters on bar 5
        strategy = SimpleStrategy(entry_bar=5)

        # Run backtest
        engine = BacktestEngine(
            strategy=strategy,
            data_feed=feed,
            broker=broker,
            risk_manager=risk_manager,
        )
        results = engine.run()

        # Verify:
        # - Position opened on bar 5
        # - Position closed on bar 15 (5 + 10 = 15)
        # - No position after bar 15

        # Get final position
        final_position = broker.get_position("TEST")

        # Should be flat (position closed by time exit)
        assert final_position.quantity == 0.0

        # Check that we actually had trades
        trades = broker.trade_tracker.get_all_trades()
        assert len(trades) > 0

        # Should have at least entry and exit
        assert any(t.side.name == "BUY" for t in trades)
        assert any(t.side.name == "SELL" for t in trades)


class TestPriceBasedStopLossIntegration:
    """Integration tests for PriceBasedStopLoss rule."""

    def test_stop_loss_exits_on_breach(self, temp_data_file):
        """Test that stop loss exits when price drops below threshold."""
        # Create downtrending data
        feed = PolarsDataFeed(temp_data_file, asset_id="TEST")
        broker = SimulationBroker(initial_cash=100_000)

        # Create risk manager with stop loss at 95 (entry ~100, will hit stop)
        risk_manager = RiskManager()
        risk_manager.add_rule(PriceBasedStopLoss(stop_loss_price=95.0))

        # Create strategy that enters on bar 5
        strategy = SimpleStrategy(entry_bar=5)

        # Run backtest
        engine = BacktestEngine(
            strategy=strategy,
            data_feed=feed,
            broker=broker,
            risk_manager=risk_manager,
        )
        results = engine.run()

        # Verify:
        # - Position opened on bar 5 (price ~100)
        # - Position closed when price hits 95 (should be in downtrend phase)
        # - Final position is flat

        final_position = broker.get_position("TEST")
        assert final_position.quantity == 0.0

        trades = broker.trade_tracker.get_all_trades()
        assert len(trades) > 0
        assert any(t.side.name == "BUY" for t in trades)
        assert any(t.side.name == "SELL" for t in trades)


class TestPriceBasedTakeProfitIntegration:
    """Integration tests for PriceBasedTakeProfit rule."""

    def test_take_profit_exits_on_target(self, temp_data_file):
        """Test that take profit exits when price reaches target."""
        # Create uptrending data
        feed = PolarsDataFeed(temp_data_file, asset_id="TEST")
        broker = SimulationBroker(initial_cash=100_000)

        # Create risk manager with take profit at 115 (entry ~100, will hit TP)
        risk_manager = RiskManager()
        risk_manager.add_rule(PriceBasedTakeProfit(take_profit_price=115.0))

        # Create strategy that enters on bar 5
        strategy = SimpleStrategy(entry_bar=5)

        # Run backtest
        engine = BacktestEngine(
            strategy=strategy,
            data_feed=feed,
            broker=broker,
            risk_manager=risk_manager,
        )
        results = engine.run()

        # Verify:
        # - Position opened on bar 5 (price ~100)
        # - Position closed when price hits 115 (should be in uptrend phase)
        # - Final position is flat

        final_position = broker.get_position("TEST")
        assert final_position.quantity == 0.0

        trades = broker.trade_tracker.get_all_trades()
        assert len(trades) > 0
        assert any(t.side.name == "BUY" for t in trades)
        assert any(t.side.name == "SELL" for t in trades)


class TestCombinedRulesIntegration:
    """Integration tests for multiple rules working together."""

    def test_multiple_rules_work_together(self, temp_data_file):
        """Test that multiple rules can be combined."""
        feed = PolarsDataFeed(temp_data_file, asset_id="TEST")
        broker = SimulationBroker(initial_cash=100_000)

        # Create risk manager with all three rules
        risk_manager = RiskManager()
        risk_manager.add_rule(TimeBasedExit(max_bars=20))
        risk_manager.add_rule(PriceBasedStopLoss(stop_loss_price=92.0))
        risk_manager.add_rule(PriceBasedTakeProfit(take_profit_price=118.0))

        strategy = SimpleStrategy(entry_bar=5)

        engine = BacktestEngine(
            strategy=strategy,
            data_feed=feed,
            broker=broker,
            risk_manager=risk_manager,
        )
        results = engine.run()

        # Verify that position was closed by one of the rules
        final_position = broker.get_position("TEST")
        assert final_position.quantity == 0.0

        trades = broker.trade_tracker.get_all_trades()
        assert len(trades) > 0

    def test_stop_loss_has_higher_priority_than_take_profit(self, temp_data_file):
        """Test that stop loss (priority 10) is checked before take profit (priority 8)."""
        feed = PolarsDataFeed(temp_data_file, asset_id="TEST")
        broker = SimulationBroker(initial_cash=100_000)

        # Create risk manager with both rules
        risk_manager = RiskManager()
        stop_loss_rule = PriceBasedStopLoss(stop_loss_price=95.0)
        take_profit_rule = PriceBasedTakeProfit(take_profit_price=115.0)

        risk_manager.add_rule(stop_loss_rule)
        risk_manager.add_rule(take_profit_rule)

        # Verify priorities
        assert stop_loss_rule.priority > take_profit_rule.priority

        strategy = SimpleStrategy(entry_bar=5)

        engine = BacktestEngine(
            strategy=strategy,
            data_feed=feed,
            broker=broker,
            risk_manager=risk_manager,
        )
        results = engine.run()

        # Should exit (either by stop or take profit depending on price action)
        final_position = broker.get_position("TEST")
        assert final_position.quantity == 0.0
