"""Session expiration follows actual calendar clocks without fabricating market bars."""

import json
from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy
from ml4t.backtest.config import ExecutionMode
from ml4t.backtest.types import OrderStatus


def _frame(times):
    return pl.DataFrame(
        {"timestamp": times, "asset": ["TEST"] * len(times), "close": [100.0] * len(times)}
    )


@pytest.mark.parametrize("warn", [False, True])
def test_explicit_order_session_bounds_expire_only_the_partial_remainder_and_keep_caller_copy(
    caplog,
    warn,
):
    from ml4t.backtest.execution import VolumeParticipationLimit

    times = [
        datetime(2025, 7, 2, 15, 59, tzinfo=UTC),
        datetime(2025, 7, 2, 16, tzinfo=UTC),
        datetime(2025, 7, 2, 16, 1, tzinfo=UTC),
    ]
    orders = []

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                orders.append(broker.submit_order("TEST", 3))
            with pytest.raises(RuntimeError, match="before the first market event"):
                broker.configure_session_cancellation(bounds={})

    engine = Engine(
        DataFeed(
            prices_df=_frame(times).with_columns(pl.lit(2.0).alias("volume")),
            feed_spec={"timestamp_semantics": "bar_close"},
        ),
        Submit(),
        BacktestConfig(
            calendar="NYSE", execution_mode=ExecutionMode.NEXT_BAR, cancel_policy="session_end"
        ),
        execution_limits=VolumeParticipationLimit(0.5),
    )
    opened = datetime(2025, 7, 2, 13, 30, tzinfo=UTC)
    bounds = {times[0].date(): (opened, times[1])}
    engine.broker.configure_session_cancellation(bounds=bounds, warn_on_cancel=warn)
    bounds[times[0].date()] = (opened, times[2])
    # A rejected replacement must not discard the previously validated configuration.
    with pytest.raises(ValueError, match="within their declared session"):
        engine.broker.configure_session_cancellation(
            bounds={times[0].date(): (opened, datetime(2025, 7, 2, 23, tzinfo=UTC))}
        )
    result = engine.run()
    assert result.to_fills_dataframe()["quantity"].to_list() == [1.0]
    assert orders[0].status is OrderStatus.CANCELLED and orders[0].filled_quantity == 1.0
    assert orders[0].cancelled_at == times[1]
    warnings = [
        record.message for record in caplog.records if "cancelled at session end" in record.message
    ]
    if warn:
        assert len(warnings) == 1 and "1.0 filled, 2.0 remaining" in warnings[0]
    else:
        assert warnings == []


@pytest.mark.parametrize(
    "policy,expected_status",
    [("never", OrderStatus.FILLED), ("session_end", OrderStatus.CANCELLED)],
)
def test_session_cancel_prevents_an_old_order_filling_at_the_next_sessions_open(
    policy, expected_status
):
    times = [datetime(2025, 1, 2, 20, 55, tzinfo=UTC), datetime(2025, 1, 3, 14, 30, tzinfo=UTC)]
    orders = []

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                orders.append(broker.submit_order("TEST", 1))

    result = Engine(
        DataFeed(prices_df=_frame(times), feed_spec={"timestamp_semantics": "event_time"}),
        Submit(),
        BacktestConfig(
            calendar="NYSE", execution_mode=ExecutionMode.NEXT_BAR, cancel_policy=policy
        ),
    ).run()
    assert orders[0].status is expected_status
    assert result.to_equity_dataframe()["timestamp"].to_list() == times
    assert result.to_fills_dataframe()["quantity"].to_list() == ([1.0] if policy == "never" else [])
    if policy == "session_end":
        assert orders[0].cancelled_at == datetime(2025, 1, 2, 21, tzinfo=UTC)
        assert orders[0].cancellation_reason == "session_end"


def test_partial_order_can_fill_at_the_close_then_only_its_remainder_expires():
    from ml4t.backtest.execution import VolumeParticipationLimit

    times = [datetime(2025, 7, 3, 16, 55, tzinfo=UTC), datetime(2025, 7, 3, 17, tzinfo=UTC)]
    orders = []

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                orders.append(broker.submit_order("TEST", 10))

    result = Engine(
        DataFeed(
            prices_df=_frame(times).with_columns(pl.lit(10.0).alias("volume")),
            feed_spec={"timestamp_semantics": "bar_close"},
        ),
        Submit(),
        BacktestConfig(
            calendar="NYSE", execution_mode=ExecutionMode.NEXT_BAR, cancel_policy="session_end"
        ),
        execution_limits=VolumeParticipationLimit(max_participation=0.5),
    ).run()
    assert result.to_fills_dataframe()["quantity"].to_list() == [5.0]
    assert orders[0].filled_quantity == 5 and orders[0].status is OrderStatus.CANCELLED
    assert orders[0].cancelled_at == times[-1]


def test_ending_a_run_before_the_official_close_keeps_its_open_orders():
    times = [datetime(2025, 7, 3, 16, 55, tzinfo=UTC)]
    orders = []

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            orders.append(broker.submit_order("TEST", 1))

    Engine(
        DataFeed(prices_df=_frame(times), feed_spec={"timestamp_semantics": "event_time"}),
        Submit(),
        BacktestConfig(calendar="NYSE", cancel_policy="session_end"),
    ).run()
    assert orders[0].status is OrderStatus.PENDING
    assert orders[0].cancelled_at is None


def test_sparse_cme_pre_open_expires_only_the_old_session_before_callback_visibility():
    from ml4t.specs import LifecyclePhase

    times = [datetime(2025, 7, 7, 20, 30, tzinfo=UTC), datetime(2025, 7, 7, 23, tzinfo=UTC)]
    runtime = RuntimeServices(start=times[0])
    orders = []
    trace = []

    def before(timestamp, data, context, broker):
        trace.append(orders[0].status)
        orders.append(broker.submit_order("TEST", 1))

    runtime.scheduler.register(
        before, phase=LifecyclePhase.PRE_OPEN, at="session_open", minutes_before=45
    )

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                orders.append(broker.submit_order("TEST", 1))

    result = Engine(
        DataFeed(prices_df=_frame(times), feed_spec={"timestamp_semantics": "event_time"}),
        Submit(),
        BacktestConfig(
            calendar="CME_Equity",
            execution_mode=ExecutionMode.NEXT_BAR,
            cancel_policy="session_end",
        ),
        runtime=runtime,
    ).run()
    assert trace == [OrderStatus.CANCELLED]
    assert [o.status for o in orders] == [OrderStatus.CANCELLED, OrderStatus.FILLED]
    assert orders[0].cancelled_at == datetime(2025, 7, 7, 21, tzinfo=UTC)
    assert result.to_fills_dataframe()["timestamp"].to_list() == [times[-1]]


def test_pre_open_before_the_previous_close_does_not_expire_a_still_live_order_early():
    from ml4t.specs import LifecyclePhase

    times = [datetime(2025, 7, 7, 20, 25, tzinfo=UTC), datetime(2025, 7, 7, 23, tzinfo=UTC)]
    runtime = RuntimeServices(start=times[0])
    orders = []
    trace = []
    runtime.scheduler.register(
        lambda *_args: trace.append(orders[0].status),
        phase=LifecyclePhase.PRE_OPEN,
        at="session_open",
        minutes_before=90,
    )

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                orders.append(broker.submit_order("TEST", 1))

    Engine(
        DataFeed(prices_df=_frame(times), feed_spec={"timestamp_semantics": "event_time"}),
        Submit(),
        BacktestConfig(calendar="CME_Equity", cancel_policy="session_end"),
        runtime=runtime,
    ).run()
    assert trace == [OrderStatus.PENDING]
    assert orders[0].status is OrderStatus.CANCELLED


@pytest.mark.parametrize(
    "calendar,role", [(None, "event_time"), ("NYSE", None), ("NYSE", "session_label")]
)
def test_session_cancel_requires_a_calendar_and_declared_physical_instants(calendar, role):
    times = [datetime(2025, 1, 2, 21, tzinfo=UTC)]

    class Observe(Strategy):
        def on_data(self, *args):
            pass

    with pytest.raises(ValueError, match="calendar|timestamp_semantics"):
        Engine(
            DataFeed(prices_df=_frame(times), feed_spec={"timestamp_semantics": role}),
            Observe(),
            BacktestConfig(calendar=calendar, cancel_policy="session_end"),
        )


def test_cancel_policy_survives_strict_config_json_and_yaml_round_trips(tmp_path):
    config = BacktestConfig(cancel_policy="session_end")
    assert config.to_dict()["execution"]["cancel_policy"] == "session_end"
    assert BacktestConfig.from_dict(config.to_dict(), strict=True).cancel_policy == "session_end"
    assert (
        BacktestConfig.from_dict(
            json.loads(json.dumps(config.to_dict())), strict=True
        ).cancel_policy
        == "session_end"
    )
    path = tmp_path / "cancel-policy.yaml"
    config.to_yaml(path)
    assert BacktestConfig.from_yaml(path).cancel_policy == "session_end"


@pytest.mark.parametrize(
    "role,expected", [("event_time", OrderStatus.CANCELLED), ("bar_close", OrderStatus.FILLED)]
)
def test_continuous_midnight_open_and_previous_close_have_distinct_expiration_ordering(
    role, expected
):
    times = [datetime(2025, 1, 1, 23, 59, tzinfo=UTC), datetime(2025, 1, 2, tzinfo=UTC)]
    orders = []

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                orders.append(broker.submit_order("TEST", 1))

    Engine(
        DataFeed(prices_df=_frame(times), feed_spec={"timestamp_semantics": role}),
        Submit(),
        BacktestConfig(
            calendar="crypto", execution_mode=ExecutionMode.NEXT_BAR, cancel_policy="session_end"
        ),
    ).run()
    assert orders[0].status is expected


def test_current_bar_moc_fills_before_expiration_at_the_actual_session_close():
    from ml4t.backtest.types import OrderType

    times = [datetime(2025, 1, 2, 20, 55, tzinfo=UTC), datetime(2025, 1, 2, 21, tzinfo=UTC)]
    orders = []

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[-1]:
                orders.append(broker.submit_order("TEST", 1, order_type=OrderType.MOC))

    result = Engine(
        DataFeed(prices_df=_frame(times), feed_spec={"timestamp_semantics": "bar_close"}),
        Submit(),
        BacktestConfig(
            calendar="NYSE", execution_mode=ExecutionMode.NEXT_BAR, cancel_policy="session_end"
        ),
    ).run()
    assert result.to_fills_dataframe()["timestamp"].to_list() == [times[-1]]
    assert orders[0].status is OrderStatus.FILLED and orders[0].cancelled_at is None


def test_upcoming_pre_open_orders_cannot_fill_on_the_previous_sessions_closing_bar():
    from ml4t.specs import LifecyclePhase

    times = [
        datetime(2025, 7, 7, 20, 25, tzinfo=UTC),
        datetime(2025, 7, 7, 21, tzinfo=UTC),
        datetime(2025, 7, 7, 23, tzinfo=UTC),
    ]
    runtime = RuntimeServices(start=times[0])
    runtime.scheduler.register(
        lambda _timestamp, _data, _context, broker: broker.submit_order("TEST", 1),
        phase=LifecyclePhase.PRE_OPEN,
        at="session_open",
        minutes_before=90,
    )

    class Observe(Strategy):
        def on_data(self, *args):
            pass

    result = Engine(
        DataFeed(prices_df=_frame(times), feed_spec={"timestamp_semantics": "bar_close"}),
        Observe(),
        BacktestConfig(calendar="CME_Equity", execution_mode=ExecutionMode.NEXT_BAR),
        runtime=runtime,
    ).run()
    assert result.to_fills_dataframe()["timestamp"].to_list() == [times[-1]]


def test_session_cancel_rejects_a_non_session_order_before_any_account_mutation():
    timestamp = datetime(2025, 1, 4, 15, tzinfo=UTC)

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            broker.submit_order("TEST", 1)

    engine = Engine(
        DataFeed(prices_df=_frame([timestamp]), feed_spec={"timestamp_semantics": "event_time"}),
        Submit(),
        BacktestConfig(
            calendar="NYSE", cancel_policy="session_end", execution_mode=ExecutionMode.SAME_BAR
        ),
    )
    with pytest.raises(ValueError, match="session"):
        engine.run()
    assert engine.broker.get_cash() == 100000 and engine.broker.positions == {}
    assert engine.broker.orders == []


def test_catching_a_non_session_order_error_does_not_consume_the_next_valid_order_identifier():
    times = [datetime(2025, 1, 4, 15, tzinfo=UTC), datetime(2025, 1, 6, 15, tzinfo=UTC)]
    ids = []

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                with pytest.raises(ValueError, match="session"):
                    broker.submit_order("TEST", 1)
            else:
                ids.append(broker.submit_order("TEST", 1).order_id)

    Engine(
        DataFeed(prices_df=_frame(times), feed_spec={"timestamp_semantics": "event_time"}),
        Submit(),
        BacktestConfig(calendar="NYSE", cancel_policy="session_end"),
    ).run()
    assert ids == ["ORD-1"]


def test_declared_context_clock_can_expire_an_order_without_an_executable_closing_quote():
    times = [datetime(2025, 1, 2, 20, 55, tzinfo=UTC), datetime(2025, 1, 2, 21, tzinfo=UTC)]
    orders = []

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                orders.append(broker.submit_order("TEST", 1))
            else:
                assert data == {} and broker.get_mark_price("TEST") is None

    feed = DataFeed(
        prices_df=_frame(times[:1]),
        context_df=pl.DataFrame({"timestamp": times[1:], "session_clock": [True]}),
        feed_spec={"timestamp_semantics": "event_time"},
    )
    result = Engine(
        feed, Submit(), BacktestConfig(calendar="NYSE", cancel_policy="session_end")
    ).run()
    assert orders[0].status is OrderStatus.CANCELLED and orders[0].cancelled_at == times[-1]
    assert result.to_fills_dataframe().is_empty()
    assert feed.prices["timestamp"].to_list() == times[:1]
