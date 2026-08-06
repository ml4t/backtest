"""End-to-end tests for strategy work that must run before position risk."""

from datetime import datetime, timedelta

import polars as pl

from ml4t.backtest import (
    BacktestConfig,
    Broker,
    DataFeed,
    Engine,
    ExecutionMode,
    OrderType,
    StopLoss,
    Strategy,
    TrailingStop,
)
from ml4t.backtest.config import ExecutionPrice, WaterMarkSource


class OpeningTargetWithStop(Strategy):
    """Enter at the session open; SAME_BAR immediate mode protects the entry bar."""

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
        if broker.get_position("SPY") is None and not broker.get_pending_orders("SPY"):
            broker.submit_order("SPY", 10)

    def on_data(self, timestamp, data, context, broker) -> None:
        self._record("on_data", timestamp, broker)


class ExplicitPreRiskPyramiding(Strategy):
    """Submit an additional lot on every bar without a position guard."""

    def on_before_risk(self, timestamp, data, context, broker) -> None:
        broker.submit_order("SPY", 10)

    def on_data(self, timestamp, data, context, broker) -> None:
        pass


class PendingAwareLimitEntry(Strategy):
    """Keep one untriggered limit order while the position remains flat."""

    def on_before_risk(self, timestamp, data, context, broker) -> None:
        if broker.get_position("SPY") is None and not broker.get_pending_orders("SPY"):
            broker.submit_order(
                "SPY",
                10,
                order_type=OrderType.LIMIT,
                limit_price=50.0,
            )

    def on_data(self, timestamp, data, context, broker) -> None:
        pass


class ExitFundedEntry(Strategy):
    """Exit one asset under risk before funding a prior-bar entry in another."""

    def on_data(self, timestamp, data, context, broker) -> None:
        if timestamp.day == 3:
            broker.set_position_rules(StopLoss(pct=0.05), asset="AAPL")
            broker.submit_order("AAPL", 90)
        elif timestamp.day == 4:
            broker.submit_order("GOOGL", 90)


class ExitFundedPreRiskEntry(Strategy):
    """Keep an unaffordable pre-risk entry pending until a risk exit funds it."""

    def on_before_risk(self, timestamp, data, context, broker) -> None:
        if (
            timestamp.day == 4
            and broker.get_position("GOOGL") is None
            and not broker.get_pending_orders("GOOGL")
        ):
            broker.submit_order("GOOGL", 90)

    def on_data(self, timestamp, data, context, broker) -> None:
        if timestamp.day == 3:
            broker.set_position_rules(StopLoss(pct=0.05), asset="AAPL")
            broker.submit_order("AAPL", 90)


class FlatLimitThenOrdinaryMarket(Strategy):
    """Open through on_data while an older pre-risk limit remains pending."""

    def __init__(self) -> None:
        self.visible_quantities: list[tuple[int, float]] = []

    def on_before_risk(self, timestamp, data, context, broker) -> None:
        position = broker.get_position("SPY")
        self.visible_quantities.append(
            (timestamp.day, 0.0 if position is None else position.quantity)
        )
        if timestamp.day == 3:
            broker.submit_order(
                "SPY",
                10,
                order_type=OrderType.LIMIT,
                limit_price=50.0,
            )

    def on_data(self, timestamp, data, context, broker) -> None:
        if timestamp.day == 3:
            broker.submit_order("SPY", 10)


class LatePricePreRiskEntry(Strategy):
    """Keep a pre-risk market order pending until its asset first has a price."""

    def __init__(self) -> None:
        self.visible_quantities: list[tuple[int, float]] = []

    def on_before_risk(self, timestamp, data, context, broker) -> None:
        position = broker.get_position("SPY")
        self.visible_quantities.append(
            (timestamp.day, 0.0 if position is None else position.quantity)
        )
        if timestamp.day == 3:
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
        BacktestConfig(
            execution_mode=ExecutionMode.NEXT_BAR,
            next_bar_queue_shadow_validation=True,
        ),
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


def test_next_bar_pre_risk_does_not_evaluate_new_position_until_following_bar() -> None:
    prices = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 8, day) for day in (3, 4, 5)],
            "asset": ["SPY"] * 3,
            "open": [100.0] * 3,
            "high": [101.0] * 3,
            "low": [99.0, 94.0, 94.0],
            "close": [100.0] * 3,
            "volume": [1_000_000.0] * 3,
        }
    )

    result = Engine(
        DataFeed(prices_df=prices),
        OpeningTargetWithStop(),
        BacktestConfig(execution_mode=ExecutionMode.NEXT_BAR),
    ).run()

    assert [(fill.side.value, fill.timestamp.day) for fill in result.fills] == [
        ("buy", 4),
        ("sell", 5),
    ]


def test_next_bar_pre_risk_excludes_entry_bar_extreme_from_trailing_watermark() -> None:
    prices = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 8, day) for day in (3, 4, 5)],
            "asset": ["SPY"] * 3,
            "open": [100.0] * 3,
            "high": [100.0, 200.0, 100.0],
            "low": [100.0, 95.0, 96.0],
            "close": [100.0] * 3,
            "volume": [1_000_000.0] * 3,
        }
    )
    engine = Engine(
        DataFeed(prices_df=prices),
        OpeningTargetWithTrailingStop(),
        BacktestConfig(
            execution_mode=ExecutionMode.NEXT_BAR,
            trail_hwm_source=WaterMarkSource.BAR_EXTREME,
        ),
    )

    result = engine.run()

    assert [(fill.side.value, fill.timestamp.day) for fill in result.fills] == [("buy", 4)]
    assert engine.broker.get_position("SPY").high_water_mark == 100.0


def test_next_bar_exit_first_preserves_exit_funded_entry() -> None:
    rows = []
    for day in (3, 4, 5):
        for asset in ("AAPL", "GOOGL"):
            low = 94.0 if day == 5 and asset == "AAPL" else 100.0
            rows.append(
                {
                    "timestamp": datetime(2026, 8, day),
                    "asset": asset,
                    "open": 100.0,
                    "high": 100.0,
                    "low": low,
                    "close": 100.0,
                    "volume": 1_000_000.0,
                }
            )

    result = Engine(
        DataFeed(prices_df=pl.DataFrame(rows)),
        ExitFundedEntry(),
        BacktestConfig(
            initial_cash=10_000.0,
            execution_mode=ExecutionMode.NEXT_BAR,
        ),
    ).run()

    assert result.rejected_orders == []
    assert [(fill.asset, fill.side.value, fill.timestamp.day) for fill in result.fills] == [
        ("AAPL", "buy", 4),
        ("AAPL", "sell", 5),
        ("GOOGL", "buy", 5),
    ]


def test_next_bar_pending_limit_guard_does_not_duplicate_intent() -> None:
    engine = Engine(
        DataFeed(prices_df=_daily_prices()),
        PendingAwareLimitEntry(),
        BacktestConfig(execution_mode=ExecutionMode.NEXT_BAR),
    )

    result = engine.run()

    assert result.fills == []
    assert len(engine.broker.orders) == 1
    assert len(engine.broker.get_pending_orders("SPY")) == 1


def test_next_bar_pre_risk_entry_can_use_same_bar_exit_proceeds() -> None:
    rows = []
    for day in (3, 4, 5):
        for asset in ("AAPL", "GOOGL"):
            low = 94.0 if day == 5 and asset == "AAPL" else 100.0
            rows.append(
                {
                    "timestamp": datetime(2026, 8, day),
                    "asset": asset,
                    "open": 100.0,
                    "high": 100.0,
                    "low": low,
                    "close": 100.0,
                    "volume": 1_000_000.0,
                }
            )

    engine = Engine(
        DataFeed(prices_df=pl.DataFrame(rows)),
        ExitFundedPreRiskEntry(),
        BacktestConfig(
            initial_cash=10_000.0,
            execution_mode=ExecutionMode.NEXT_BAR,
        ),
    )
    result = engine.run()

    assert result.rejected_orders == []
    assert [(fill.asset, fill.side.value, fill.timestamp.day) for fill in result.fills] == [
        ("AAPL", "buy", 4),
        ("AAPL", "sell", 5),
        ("GOOGL", "buy", 5),
    ]
    assert engine.broker.get_position("GOOGL").quantity == 90.0


def test_pre_risk_limit_is_rechecked_for_flatness_before_early_fill() -> None:
    prices = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 8, day) for day in (3, 4, 5)],
            "asset": ["SPY"] * 3,
            "open": [100.0, 100.0, 50.0],
            "high": [100.0, 100.0, 50.0],
            "low": [100.0, 100.0, 50.0],
            "close": [100.0, 100.0, 50.0],
            "volume": [1_000_000.0] * 3,
        }
    )
    strategy = FlatLimitThenOrdinaryMarket()
    engine = Engine(
        DataFeed(prices_df=prices),
        strategy,
        BacktestConfig(execution_mode=ExecutionMode.NEXT_BAR),
    )

    result = engine.run()

    assert strategy.visible_quantities == [(3, 0.0), (4, 0.0), (5, 10.0)]
    assert [(fill.timestamp.day, fill.quantity) for fill in result.fills] == [
        (4, 10.0),
        (5, 10.0),
    ]
    assert engine.broker.get_position("SPY").quantity == 20.0


def test_aged_pre_risk_market_entry_uses_queue_shadow_validation() -> None:
    prices = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 8, day) for day in (3, 4, 5)],
            "asset": ["AAPL", "AAPL", "SPY"],
            "open": [100.0] * 3,
            "high": [100.0] * 3,
            "low": [100.0] * 3,
            "close": [100.0] * 3,
            "volume": [1_000_000.0] * 3,
        }
    )
    strategy = LatePricePreRiskEntry()
    result = Engine(
        DataFeed(prices_df=prices),
        strategy,
        BacktestConfig(
            execution_mode=ExecutionMode.NEXT_BAR,
            next_bar_queue_shadow_validation=True,
        ),
    ).run()

    assert [(fill.asset, fill.timestamp.day) for fill in result.fills] == [("SPY", 5)]
    assert strategy.visible_quantities == [(3, 0.0), (4, 0.0), (5, 10.0)]
