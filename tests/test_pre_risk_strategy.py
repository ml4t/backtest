"""End-to-end tests for strategy work that must run before position risk."""

from datetime import datetime, timedelta

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


class GuardedPreRiskEntry(Strategy):
    """Enter only while no position exists and record callback-visible state."""

    def __init__(self) -> None:
        self.trace: list[tuple[str, int, float, int]] = []

    def _record(self, phase: str, timestamp: datetime, broker: Broker) -> None:
        position = broker.get_position("SPY")
        self.trace.append(
            (
                phase,
                timestamp.day,
                0.0 if position is None else position.quantity,
                len(broker.get_pending_orders("SPY")),
            )
        )

    def on_before_risk(self, timestamp, data, context, broker) -> None:
        self._record("before_risk", timestamp, broker)
        if broker.get_position("SPY") is None:
            broker.submit_order("SPY", 10)

    def on_data(self, timestamp, data, context, broker) -> None:
        self._record("on_data", timestamp, broker)


class ExplicitPreRiskPyramiding(Strategy):
    """Submit an additional lot on every bar without a position guard."""

    def on_before_risk(self, timestamp, data, context, broker) -> None:
        broker.submit_order("SPY", 10)

    def on_data(self, timestamp, data, context, broker) -> None:
        pass


def _daily_prices(days: int = 3) -> pl.DataFrame:
    start = datetime(2026, 8, 3)
    timestamps = [start + timedelta(days=offset) for offset in range(days)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset": ["SPY"] * days,
            "open": [100.0] * days,
            "high": [101.0] * days,
            "low": [99.0] * days,
            "close": [100.0] * days,
            "volume": [1_000_000.0] * days,
        }
    )


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


def test_next_bar_pre_risk_guard_sees_filled_open_order() -> None:
    strategy = GuardedPreRiskEntry()
    result = Engine(
        DataFeed(prices_df=_daily_prices()),
        strategy,
        BacktestConfig(execution_mode=ExecutionMode.NEXT_BAR),
    ).run()

    assert [(fill.timestamp.day, fill.quantity) for fill in result.fills] == [(4, 10.0)]
    assert strategy.trace == [
        ("before_risk", 3, 0.0, 0),
        ("on_data", 3, 0.0, 1),
        ("before_risk", 4, 10.0, 0),
        ("on_data", 4, 10.0, 0),
        ("before_risk", 5, 10.0, 0),
        ("on_data", 5, 10.0, 0),
    ]


def test_same_bar_pre_risk_trace_is_stable() -> None:
    strategy = GuardedPreRiskEntry()
    result = Engine(
        DataFeed(prices_df=_daily_prices(days=2)),
        strategy,
        BacktestConfig(
            execution_mode=ExecutionMode.SAME_BAR,
            immediate_fill=False,
        ),
    ).run()

    assert [(fill.timestamp.day, fill.quantity) for fill in result.fills] == [(3, 10.0)]
    assert strategy.trace == [
        ("before_risk", 3, 0.0, 0),
        ("on_data", 3, 10.0, 0),
        ("before_risk", 4, 10.0, 0),
        ("on_data", 4, 10.0, 0),
    ]


def test_next_bar_pre_risk_allows_explicit_pyramiding() -> None:
    engine = Engine(
        DataFeed(prices_df=_daily_prices()),
        ExplicitPreRiskPyramiding(),
        BacktestConfig(execution_mode=ExecutionMode.NEXT_BAR),
    )

    result = engine.run()

    assert sum(fill.quantity for fill in result.fills) == 20.0
    assert engine.broker.get_position("SPY").quantity == 20.0
    assert len(engine.broker.get_pending_orders("SPY")) == 1
