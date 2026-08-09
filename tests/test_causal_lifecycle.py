from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from ml4t.specs import HistoricalStrategyCompatibilityError, LifecyclePhase

from ml4t.backtest import BacktestConfig, DataFeed, Engine, ExecutionMode, Strategy, callback_trace
from ml4t.backtest.broker import Broker
from ml4t.backtest.types import OrderStatus, OrderType, Position


def prices(*, bars: int = 1) -> pl.DataFrame:
    timestamps = [datetime(2026, 8, 3 + index) for index in range(bars)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset": ["SPY"] * bars,
            "open": [100.0 + index for index in range(bars)],
            "high": [105.0 + index for index in range(bars)],
            "low": [80.0 + index for index in range(bars)],
            "close": [90.0 + index for index in range(bars)],
            "volume": [1_000_000.0] * bars,
        }
    )


class TraceStrategy(Strategy):
    def __init__(self) -> None:
        self.trace: list[str] = []
        self.prepare_config = None

    def on_prepare(self, broker, config=None) -> None:
        self.trace.append("on_prepare")
        self.prepare_config = config

    def on_start(self, broker) -> None:
        self.trace.append("on_start")

    def on_data(self, timestamp, data, context, broker) -> None:
        self.trace.append("on_data")

    def on_end(self, broker) -> None:
        self.trace.append("on_end")


def test_engine_dispatches_one_causal_versioned_callback_trace() -> None:
    strategy = TraceStrategy()
    engine = Engine(
        DataFeed(prices_df=prices(bars=2)),
        strategy,
        BacktestConfig(retain_lifecycle_history=True),
    )

    result = engine.run()

    assert strategy.trace == ["on_start", "on_prepare", "on_data", "on_data", "on_end"]
    assert strategy.prepare_config is engine.config
    assert callback_trace(engine.lifecycle_dispatcher.invocations) == (
        ("run_start", "on_start", None),
        ("causal_initialization", "on_prepare", None),
        ("market_event", "on_data", datetime(2026, 8, 3)),
        ("market_event", "on_data", datetime(2026, 8, 4)),
        ("run_end", "on_end", None),
    )
    assert engine.lifecycle_dispatcher.callback_counts[LifecyclePhase.MARKET_EVENT] == 2
    assert result.metrics["lifecycle_callback_counts"]["market_event"] == 2
    assert len(result.metrics["lifecycle_invocations"]) == 5


def test_default_lifecycle_history_is_bounded_to_counts() -> None:
    engine = Engine(DataFeed(prices_df=prices(bars=2)), TraceStrategy())

    result = engine.run()

    assert engine.lifecycle_dispatcher.invocations == []
    assert engine.lifecycle_dispatcher.callback_counts[LifecyclePhase.MARKET_EVENT] == 2
    assert result.metrics["lifecycle_callback_counts"]["market_event"] == 2
    assert result.metrics["lifecycle_invocations"] == []


def test_completed_close_cannot_create_a_same_timestamp_open_fill() -> None:
    class CloseAwareStrategy(Strategy):
        def on_data(self, timestamp, data, context, broker) -> None:
            bar = data["SPY"]
            if bar["close"] < bar["open"]:
                broker.submit_order("SPY", 10)

    result = Engine(
        DataFeed(prices_df=prices()),
        CloseAwareStrategy(),
        BacktestConfig(execution_mode=ExecutionMode.NEXT_BAR),
    ).run()

    assert result.fills == []


def test_retained_on_before_risk_override_fails_before_broker_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker_created = False

    class LegacyStrategy(Strategy):
        def on_before_risk(self, timestamp, data, context, broker) -> None:
            broker.submit_order("SPY", 10)

        def on_data(self, timestamp, data, context, broker) -> None:
            return None

    def fail_if_called(*args, **kwargs):
        nonlocal broker_created
        broker_created = True
        raise AssertionError("broker must not be created")

    monkeypatch.setattr(Broker, "from_config", fail_if_called)
    with pytest.raises(HistoricalStrategyCompatibilityError) as captured:
        Engine(DataFeed(prices_df=prices()), LegacyStrategy())

    assert not broker_created
    assert captured.value.callback == "on_before_risk"
    assert captured.value.required_phase is LifecyclePhase.PRE_OPEN


def test_future_timestamp_prepare_signature_fails_before_broker_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker_created = False

    class HistoricalStrategy(Strategy):
        def on_prepare(self, broker, timestamps, config=None) -> None:
            return None

        def on_data(self, timestamp, data, context, broker) -> None:
            return None

    def fail_if_called(*args, **kwargs):
        nonlocal broker_created
        broker_created = True
        raise AssertionError("broker must not be created")

    monkeypatch.setattr(Broker, "from_config", fail_if_called)
    with pytest.raises(HistoricalStrategyCompatibilityError) as captured:
        Engine(DataFeed(prices_df=prices()), HistoricalStrategy())

    assert not broker_created
    assert captured.value.callback == "on_prepare(timestamps)"
    assert captured.value.required_phase is LifecyclePhase.CAUSAL_INITIALIZATION


def test_unknown_lifecycle_version_fails_before_broker_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker_created = False

    def fail_if_called(*args, **kwargs):
        nonlocal broker_created
        broker_created = True
        raise AssertionError("broker must not be created")

    monkeypatch.setattr(Broker, "from_config", fail_if_called)
    with pytest.raises(ValueError, match="Unsupported lifecycle version"):
        Engine(DataFeed(prices_df=prices()), TraceStrategy(), lifecycle_version="2")
    assert not broker_created


def test_callback_failure_rolls_back_immediate_fill_and_runs_end_once() -> None:
    class FailingStrategy(Strategy):
        def __init__(self) -> None:
            self.end_calls = 0

        def on_data(self, timestamp, data, context, broker) -> None:
            broker.submit_order("SPY", 10)
            raise RuntimeError("strategy failed")

        def on_end(self, broker) -> None:
            self.end_calls += 1

    strategy = FailingStrategy()
    engine = Engine(
        DataFeed(prices_df=prices()),
        strategy,
        BacktestConfig(
            execution_mode=ExecutionMode.SAME_BAR,
            immediate_fill=True,
        ),
    )
    account = engine.broker.account
    positions = account.positions

    with pytest.raises(RuntimeError, match="strategy failed"):
        engine.run()

    assert strategy.end_calls == 1
    assert engine.broker.orders == []
    assert engine.broker.pending_orders == []
    assert engine.broker.fills == []
    assert engine.broker.positions == {}
    assert engine.broker.cash == engine.config.initial_cash
    assert engine.broker.account is account
    assert engine.broker.positions is positions
    assert engine.broker.gatekeeper.account is account
    assert engine.broker._fill_executor.account is account
    assert engine.broker._order_book.account is account
    assert engine.broker._risk_engine.account is account
    assert engine.broker._execution_engine.account is account
    assert engine.broker._portfolio_ledger.account is account
    assert engine.preopen_target_manager.account is account


def test_non_callback_engine_failure_runs_end_once(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = TraceStrategy()
    engine = Engine(DataFeed(prices_df=prices()), strategy)

    def fail_opening(timestamp):
        raise RuntimeError("opening failed")

    monkeypatch.setattr(engine.preopen_target_manager, "process_opening", fail_opening)

    with pytest.raises(RuntimeError, match="opening failed"):
        engine.run()

    assert strategy.trace == ["on_start", "on_prepare", "on_end"]


def test_failure_before_run_start_does_not_finalize_strategy() -> None:
    strategy = TraceStrategy()
    engine = Engine(
        DataFeed(prices_df=prices()),
        strategy,
        BacktestConfig(calendar="not-a-calendar"),
    )

    with pytest.raises(RuntimeError, match="not one of the registered classes"):
        engine.run()

    assert strategy.trace == []


def test_callback_failure_restores_updated_and_cancelled_pending_order() -> None:
    class FailingStrategy(Strategy):
        def on_data(self, timestamp, data, context, broker) -> None:
            pending = broker.get_pending_orders("SPY")
            if not pending:
                broker.submit_order("SPY", 10, order_type=OrderType.LIMIT, limit_price=70)
                return
            broker.update_order(pending[0].order_id, quantity=25)
            broker.cancel_order(pending[0].order_id)
            raise RuntimeError("strategy failed")

    engine = Engine(DataFeed(prices_df=prices(bars=2)), FailingStrategy())

    with pytest.raises(RuntimeError, match="strategy failed"):
        engine.run()

    pending = engine.broker.get_pending_orders("SPY")
    assert len(pending) == 1
    assert pending[0].quantity == 10
    assert pending[0].status is OrderStatus.PENDING
    assert engine.broker.orders == pending


def test_strategy_callback_cannot_replace_broker_owned_collections() -> None:
    class ReplacingStrategy(Strategy):
        def on_start(self, broker) -> None:
            broker.submit_order("SPY", 10, order_type=OrderType.LIMIT, limit_price=70)

        def on_data(self, timestamp, data, context, broker) -> None:
            broker.orders = []

    engine = Engine(DataFeed(prices_df=prices()), ReplacingStrategy())

    with pytest.raises(RuntimeError, match="direct Broker.orders assignment"):
        engine.run()

    assert len(engine.broker.orders) == 1


def test_read_only_callbacks_do_not_copy_broker_state(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(self):
        raise AssertionError("unexpected lifecycle snapshot")

    monkeypatch.setattr(Broker, "_snapshot_lifecycle_state", fail_if_called)

    Engine(DataFeed(prices_df=prices(bars=2)), TraceStrategy()).run()


def test_scoped_lifecycle_snapshot_copies_only_the_position_being_mutated() -> None:
    broker = Broker()
    broker.positions.update(
        {
            "SPY": Position("SPY", 10, 100, datetime(2026, 8, 1), current_price=100),
            "QQQ": Position("QQQ", 20, 200, datetime(2026, 8, 1), current_price=200),
        }
    )

    state = broker._snapshot_lifecycle_state(asset="SPY")

    assert set(state["positions"]) == {"SPY"}
    assert state["risk_rules"] is None
    assert state["target_intent_state"] is None
    assert state["orders_length"] == 0
    assert "orders" not in state


def test_failed_stats_reconfiguration_restores_all_asset_histories() -> None:
    class FailingStrategy(Strategy):
        def on_data(self, timestamp, data, context, broker) -> None:
            broker.configure_stats(recent_window_size=2)
            raise RuntimeError("strategy failed")

    engine = Engine(DataFeed(prices_df=prices()), FailingStrategy())
    stats = engine.broker.get_asset_stats("SPY")
    stats.recent_pnls.extend([1.0, -1.0, 2.0])
    stats.recent_wins = 2

    with pytest.raises(RuntimeError, match="strategy failed"):
        engine.run()

    restored = engine.broker.get_asset_stats("SPY")
    assert list(restored.recent_pnls) == [1.0, -1.0, 2.0]
    assert restored.recent_pnls.maxlen == 50
    assert restored.recent_wins == 2


def test_non_utc_event_timestamps_remain_feed_values() -> None:
    strategy = TraceStrategy()
    aware = prices().with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
    engine = Engine(
        DataFeed(prices_df=aware),
        strategy,
        BacktestConfig(retain_lifecycle_history=True),
    )

    engine.run()

    event_time = engine.lifecycle_dispatcher.invocations[2].event_time
    assert event_time is not None
    assert event_time == datetime(2026, 8, 3, tzinfo=UTC)
