"""End-to-end tests for strategy work that must run before position risk."""

from datetime import datetime

import polars as pl

from ml4t.backtest import (
    BacktestConfig,
    Broker,
    DataFeed,
    Engine,
    ExecutionMode,
    StopLoss,
    Strategy,
    TrailingStop,
)
from ml4t.backtest.config import ExecutionPrice, WaterMarkSource


class OpeningTargetWithStop(Strategy):
    """Enter at the session open and protect the new position on that bar."""

    def on_before_risk(self, timestamp, data, context, broker) -> None:
        if broker.get_position("SPY") is None:
            broker.set_position_rules(StopLoss(pct=0.05), asset="SPY")
            broker.submit_order("SPY", 100)

    def on_data(self, timestamp, data, context, broker) -> None:
        pass


class OpeningTargetWithTrailingStop(Strategy):
    """Enter once and retain a trailing stop across sessions."""

    def on_before_risk(self, timestamp, data, context, broker) -> None:
        if not broker.fills:
            broker.set_position_rules(TrailingStop(pct=0.05), asset="SPY")
            broker.submit_order("SPY", 100)

    def on_data(self, timestamp, data, context, broker) -> None:
        pass


def test_pre_risk_entry_can_trigger_stop_on_entry_bar():
    """A position entered at the open receives stop protection on the same bar."""
    prices = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 8, 3)],
            "asset": ["SPY"],
            "open": [100.0],
            "high": [101.0],
            "low": [94.0],
            "close": [98.0],
            "volume": [1_000_000.0],
        }
    )
    config = BacktestConfig(
        initial_cash=100_000.0,
        execution_mode=ExecutionMode.SAME_BAR,
        execution_price=ExecutionPrice.OPEN,
        immediate_fill=True,
    )

    result = Engine(DataFeed(prices_df=prices), OpeningTargetWithStop(), config).run()

    assert [(fill.side.value, fill.price) for fill in result.fills] == [
        ("buy", 100.0),
        ("sell", 95.0),
    ]
    assert result.trades[0].exit_reason == "stop_loss"


def test_entry_bar_extreme_becomes_next_bar_trailing_watermark_when_enabled():
    """Entry-bar highs are available to a lagged trail only after that bar completes."""
    prices = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 8, 3), datetime(2026, 8, 4)],
            "asset": ["SPY", "SPY"],
            "open": [100.0, 105.0],
            "high": [110.0, 106.0],
            "low": [99.0, 103.0],
            "close": [105.0, 104.0],
            "volume": [1_000_000.0, 1_000_000.0],
        }
    )
    config = BacktestConfig(
        initial_cash=100_000.0,
        execution_mode=ExecutionMode.SAME_BAR,
        execution_price=ExecutionPrice.OPEN,
        immediate_fill=True,
        trail_hwm_source=WaterMarkSource.BAR_EXTREME,
        trail_include_entry_bar_extremes=True,
    )

    result = Engine(DataFeed(prices_df=prices), OpeningTargetWithTrailingStop(), config).run()

    assert [(fill.side.value, fill.price) for fill in result.fills] == [
        ("buy", 100.0),
        ("sell", 104.5),
    ]
    assert result.fills[1].timestamp == datetime(2026, 8, 4)
    assert result.trades[0].exit_reason == "trailing_stop"


def test_entry_bar_extreme_remains_excluded_by_default():
    """The opt-in does not change the existing entry-bar watermark contract."""
    prices = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 8, 3), datetime(2026, 8, 4)],
            "asset": ["SPY", "SPY"],
            "open": [100.0, 105.0],
            "high": [110.0, 106.0],
            "low": [99.0, 103.0],
            "close": [105.0, 104.0],
            "volume": [1_000_000.0, 1_000_000.0],
        }
    )
    config = BacktestConfig(
        initial_cash=100_000.0,
        execution_mode=ExecutionMode.SAME_BAR,
        execution_price=ExecutionPrice.OPEN,
        immediate_fill=True,
        trail_hwm_source=WaterMarkSource.BAR_EXTREME,
    )

    result = Engine(DataFeed(prices_df=prices), OpeningTargetWithTrailingStop(), config).run()

    assert [(fill.side.value, fill.price) for fill in result.fills] == [("buy", 100.0)]


def test_entry_bar_extreme_option_roundtrips_and_reaches_broker():
    """Serialized configs preserve the opt-in and Broker.from_config receives it."""
    config = BacktestConfig(trail_include_entry_bar_extremes=True)

    restored = BacktestConfig.from_dict(config.to_dict())
    broker = Broker.from_config(restored)

    assert restored.trail_include_entry_bar_extremes is True
    assert broker.trail_include_entry_bar_extremes is True
