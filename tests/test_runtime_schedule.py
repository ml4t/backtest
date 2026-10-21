"""Registered callbacks follow real calendar clocks without exposing future daily bars."""

from datetime import UTC, datetime

import polars as pl
import pytest
from ml4t.specs import FeedSpec, LifecyclePhase, TimestampSemantics

from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy
from ml4t.backtest.config import CommissionType, ExecutionMode, ExecutionPrice, SlippageType


def _instant_feed(**kwargs):
    spec = FeedSpec.from_any(kwargs.pop("feed_spec", None))
    if spec.timestamp_semantics is None:
        spec = spec.with_overrides(timestamp_semantics=TimestampSemantics.EVENT_TIME)
    return DataFeed(feed_spec=spec, **kwargs)


class Cash(Strategy):
    def on_data(self, timestamp, data, context, broker):
        pass


@pytest.mark.parametrize("include_start_session", [False, True])
def test_explicit_start_session_pre_open_can_precede_trading_bound_without_warmup_callbacks(
    include_start_session,
):
    first = datetime(2025, 7, 2, 13, 31, tzinfo=UTC)
    closes = [datetime(2025, 7, day, 20 if day < 3 else 17, tzinfo=UTC) for day in [1, 2, 3]]
    runtime, trace = RuntimeServices(start=first), []

    def before(timestamp, _data, _context, _broker):
        trace.append(
            (timestamp, runtime.market.history(["A"], ["close"], bar_count=1)["close"].to_list())
        )

    runtime.scheduler.register(
        before,
        phase=LifecyclePhase.PRE_OPEN,
        at="session_open",
        minutes_before=45,
        include_start_session=include_start_session,
    )
    result = Engine(
        _instant_feed(
            prices_df=pl.DataFrame(
                {"timestamp": closes, "asset": ["A"] * 3, "close": [10.0, 20.0, 30.0]}
            ),
            feed_spec={
                "time_contract_version": 2,
                "calendar": "NYSE",
                "timestamp_semantics": "bar_close",
                "bar_label": "right",
                "data_frequency": "1d",
            },
        ),
        Cash(),
        BacktestConfig(calendar="NYSE"),
        runtime=runtime,
    ).run()
    expected = [(datetime(2025, 7, 3, 12, 45, tzinfo=UTC), [20.0])]
    if include_start_session:
        expected.insert(0, (datetime(2025, 7, 2, 12, 45, tzinfo=UTC), [10.0]))
    assert trace == expected
    assert result.to_equity_dataframe()["timestamp"].to_list() == closes[1:]
    assert runtime.start == first


def test_start_session_extension_cannot_be_used_to_run_an_early_market_callback():
    runtime = RuntimeServices()
    with pytest.raises(ValueError, match="option for pre_open"):
        runtime.scheduler.register(lambda *_: None, include_start_session=True)


@pytest.mark.parametrize("future_close", [104.0, 204.0])
def test_daily_pre_session_sees_only_prior_prices_and_close_callbacks_follow_handle_data(
    future_close,
):
    days = [
        (2024, 12, 30),
        (2024, 12, 31),
        (2025, 1, 2),
        (2025, 1, 3),
        (2025, 1, 6),
        (2025, 1, 7),
        (2025, 1, 8),
    ]
    dates = [datetime(y, m, d, 21, tzinfo=UTC) for y, m, d in days]
    prices = pl.DataFrame(
        {
            "timestamp": dates,
            "asset": ["TEST"] * 7,
            "open": [98.0 + i for i in range(7)],
            "close": [100.0 + i for i in range(7)],
        }
    )
    prices = prices.with_columns(
        pl.when(pl.col("timestamp") == dates[4])
        .then(pl.lit(future_close))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    runtime = RuntimeServices(start=datetime(2025, 1, 6, tzinfo=UTC))
    trace = []

    def before(timestamp, data, context, broker):
        assert data == {}
        assert context == {}
        trace.append(
            (
                "before",
                timestamp,
                runtime.market.history(["TEST"], ["close"], bar_count=3)["close"].to_list(),
            )
        )

    def close(timestamp, data, context, broker):
        trace.append(("close", timestamp, data["TEST"]["close"]))

    runtime.scheduler.register(
        before, phase=LifecyclePhase.PRE_OPEN, at="session_open", minutes_before=45
    )
    runtime.scheduler.register(close, phase=LifecyclePhase.CLOSE, at="session_close")

    class Daily(Strategy):
        def on_data(self, timestamp, data, context, broker):
            trace.append(("handle", timestamp, data["TEST"]["close"]))
            if timestamp == dates[4]:
                broker.submit_order("TEST", 10)

    result = Engine(
        _instant_feed(
            prices_df=prices,
            feed_spec=FeedSpec(
                time_contract_version=2,
                timestamp_semantics=TimestampSemantics.BAR_CLOSE,
                bar_label="right",
                data_frequency="1d",
                calendar="NYSE",
            ),
        ),
        Daily(),
        BacktestConfig(
            initial_cash=10000,
            calendar="NYSE",
            execution_mode=ExecutionMode.NEXT_BAR,
            execution_price=ExecutionPrice.CLOSE,
            next_bar_execution_price=ExecutionPrice.CLOSE,
            commission_type=CommissionType.NONE,
            slippage_type=SlippageType.NONE,
        ),
        runtime=runtime,
    ).run()
    assert trace[0] == ("before", datetime(2025, 1, 6, 13, 45, tzinfo=UTC), [101.0, 102.0, 103.0])
    assert [(r[0], r[1]) for r in trace] == [
        (kind, datetime(2025, 1, d, h, m, tzinfo=UTC))
        for d in [6, 7, 8]
        for kind, h, m in [("before", 13, 45), ("handle", 21, 0), ("close", 21, 0)]
    ]
    assert result.to_fills_dataframe()["timestamp"].to_list() == [dates[5]]
    assert result.to_fills_dataframe()["price"].to_list() == [105.0]
    assert result.to_equity_dataframe()["equity"].to_list() == [10000.0, 10000.0, 10010.0]


def test_missing_official_close_does_not_generate_a_close_event_from_the_last_input_bar():
    dates = [datetime(2025, 1, 2, h, tzinfo=UTC) for h in [15, 16]]
    runtime = RuntimeServices()
    calls = []
    runtime.scheduler.register(
        lambda *args: calls.append(args[0]), phase=LifecyclePhase.CLOSE, at="session_close"
    )

    class Cash(Strategy):
        def on_data(self, timestamp, data, context, broker):
            pass

    frame = pl.DataFrame({"timestamp": dates, "asset": ["TEST"] * 2, "close": [100.0, 101.0]})
    result = Engine(
        _instant_feed(prices_df=frame), Cash(), BacktestConfig(calendar="NYSE"), runtime=runtime
    ).run()
    assert calls == []
    assert result.to_equity_dataframe()["timestamp"].to_list() == dates


def test_calendar_none_rejects_session_registration_before_trading_starts():
    runtime = RuntimeServices()
    runtime.scheduler.register(lambda *_args: None, phase=LifecyclePhase.CLOSE, at="session_close")
    seen = []

    class Cash(Strategy):
        def on_start(self, broker):
            seen.append("started")

        def on_data(self, timestamp, data, context, broker):
            pass

    frame = pl.DataFrame(
        {"timestamp": [datetime(2025, 1, 2, 21, tzinfo=UTC)], "asset": ["TEST"], "close": [100.0]}
    )
    with pytest.raises(ValueError, match="calendar"):
        Engine(
            _instant_feed(prices_df=frame), Cash(), BacktestConfig(calendar=None), runtime=runtime
        ).run()
    assert seen == []


def test_registered_failure_rolls_back_cash_orders_and_records_and_restores_clock():
    runtime = RuntimeServices()
    times = [datetime(2025, 1, 2, 21, tzinfo=UTC)]
    seen = []

    def failing(timestamp, data, context, broker):
        seen.append(timestamp)
        broker.cash = 123.0
        broker.submit_order("TEST", 1)
        broker.record(value=123)
        raise RuntimeError("callback failed")

    runtime.scheduler.register(failing, at="session_close", phase=LifecyclePhase.CLOSE)
    engine = Engine(
        _instant_feed(
            prices_df=pl.DataFrame({"timestamp": times, "asset": ["TEST"], "close": [100.0]})
        ),
        Cash(),
        BacktestConfig(initial_cash=10000, calendar="NYSE"),
        runtime=runtime,
    )
    with pytest.raises(RuntimeError, match="callback failed"):
        engine.run()
    assert seen == times
    assert engine.broker.cash == 10000
    assert engine.broker.orders == []
    assert engine.broker._recorded_variables() is None
    assert engine.broker._current_time == times[0]
    with pytest.raises(RuntimeError, match="active run"):
        _ = runtime.market


def test_pre_open_orders_wait_for_actual_quote_even_with_immediate_fills_enabled():
    runtime = RuntimeServices()
    filled_in_callback = []

    def before(timestamp, data, context, broker):
        order = broker.submit_order("TEST", 1)
        filled_in_callback.append((len(broker.fills), order.created_at))

    runtime.scheduler.register(before, at="session_open", phase=LifecyclePhase.PRE_OPEN)
    instant = datetime(2025, 1, 2, 21, tzinfo=UTC)
    result = Engine(
        _instant_feed(
            prices_df=pl.DataFrame({"timestamp": [instant], "asset": ["TEST"], "close": [100.0]})
        ),
        Cash(),
        BacktestConfig(
            initial_cash=10000,
            calendar="NYSE",
            execution_mode=ExecutionMode.SAME_BAR,
            immediate_fill=True,
            execution_price=ExecutionPrice.CLOSE,
            commission_type=CommissionType.NONE,
            slippage_type=SlippageType.NONE,
        ),
        runtime=runtime,
    ).run()
    assert filled_in_callback == [(0, datetime(2025, 1, 2, 14, 30, tzinfo=UTC))]
    assert result.to_fills_dataframe()["timestamp"].to_list() == [instant]
    assert result.to_fills_dataframe()["price"].to_list() == [100.0]
    assert result.to_equity_dataframe()["timestamp"].to_list() == [instant]


@pytest.mark.parametrize(
    "dates,opens",
    [
        (
            [datetime(2025, 3, 7, 21, tzinfo=UTC), datetime(2025, 3, 10, 20, tzinfo=UTC)],
            [datetime(2025, 3, 7, 14, 30, tzinfo=UTC), datetime(2025, 3, 10, 13, 30, tzinfo=UTC)],
        ),
        ([datetime(2025, 7, 3, 17, tzinfo=UTC)], [datetime(2025, 7, 3, 13, 30, tzinfo=UTC)]),
    ],
)
def test_dst_and_early_close_come_from_calendar_and_registration_order_is_stable(dates, opens):
    runtime = RuntimeServices()
    trace = []
    for name in ["first", "second"]:
        runtime.scheduler.register(
            lambda t, _d, _c, _b, name=name: trace.append((name, t)),
            at="session_open",
            phase=LifecyclePhase.PRE_OPEN,
        )
    runtime.scheduler.register(
        lambda t, *_args: trace.append(("close", t)), at="session_close", phase=LifecyclePhase.CLOSE
    )
    frame = pl.DataFrame(
        {"timestamp": dates, "asset": ["TEST"] * len(dates), "close": [100.0] * len(dates)}
    )
    Engine(
        _instant_feed(prices_df=frame), Cash(), BacktestConfig(calendar="NYSE"), runtime=runtime
    ).run()
    assert trace == [
        item
        for opened, closed in zip(opens, dates, strict=True)
        for item in [("first", opened), ("second", opened), ("close", closed)]
    ]


@pytest.mark.parametrize("override,expected", [(None, 190.0), (ExecutionPrice.CLOSE, 200.0)])
def test_next_bar_close_is_explicit_and_does_not_change_native_next_open_default(
    override, expected
):
    times = [datetime(2025, 1, 2, 21, tzinfo=UTC), datetime(2025, 1, 3, 21, tzinfo=UTC)]
    frame = pl.DataFrame(
        {"timestamp": times, "asset": ["TEST"] * 2, "open": [90.0, 190.0], "close": [100.0, 200.0]}
    )

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("TEST", 1)

    config = BacktestConfig(
        initial_cash=10000,
        calendar=None,
        execution_mode=ExecutionMode.NEXT_BAR,
        execution_price=ExecutionPrice.CLOSE,
        next_bar_execution_price=override,
        commission_type=CommissionType.NONE,
        slippage_type=SlippageType.NONE,
    )
    restored = BacktestConfig.from_dict(config.to_dict(), strict=True)
    result = Engine(_instant_feed(prices_df=frame), Buy(), restored).run()
    assert result.to_fills_dataframe()["price"].to_list() == [expected]
    assert result.to_fills_dataframe()["timestamp"].to_list() == [times[1]]
    assert result.to_equity_dataframe()["equity"].to_list() == [10000.0, 10200.0 - expected]


def test_bar_cadence_counts_sessions_instead_of_treating_every_bar_as_session_one():
    from ml4t.backtest.execution.schedule import RebalanceSchedule

    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [2, 3, 6]]
    runtime = RuntimeServices()
    seen = []
    runtime.scheduler.register(
        lambda t, *_args: seen.append(t), at="bar", schedule=RebalanceSchedule.fixed_n_sessions(2)
    )
    frame = pl.DataFrame({"timestamp": times, "asset": ["TEST"] * 3, "close": [100.0] * 3})
    Engine(
        _instant_feed(prices_df=frame), Cash(), BacktestConfig(calendar="NYSE"), runtime=runtime
    ).run()
    assert seen == [times[0], times[2]]


def test_run_scoped_registrations_do_not_accumulate_when_runtime_is_reused():
    runtime = RuntimeServices()
    seen = []
    runtime.scheduler.register(lambda *_args: seen.append("static"))

    class Observe(Cash):
        def on_start(self, broker):
            runtime.scheduler.register(lambda *_args: seen.append("run"))

    frame = pl.DataFrame(
        {"timestamp": [datetime(2025, 1, 2, 21, tzinfo=UTC)], "asset": ["TEST"], "close": [100.0]}
    )
    for _ in range(2):
        Engine(
            _instant_feed(prices_df=frame),
            Observe(),
            BacktestConfig(calendar=None),
            runtime=runtime,
        ).run()
    assert seen == ["static", "run", "static", "run"]


def test_pre_open_history_respects_publication_at_the_callback_instant():
    now = datetime(2025, 1, 3, 14, 30, tzinfo=UTC)
    times = [datetime(2025, 1, 2, 21, tzinfo=UTC), datetime(2025, 1, 3, 21, tzinfo=UTC)]
    frame = pl.DataFrame(
        {
            "timestamp": times,
            "asset": ["TEST"] * 2,
            "close": [100.0, 101.0],
            "feature": [3.0, 99.0],
            "known": [now, times[1]],
        }
    )
    runtime = RuntimeServices(
        start=datetime(2025, 1, 3, tzinfo=UTC), known_at_cols={"feature": "known"}
    )
    seen = []

    def before(timestamp, data, context, broker):
        seen.append(runtime.market.history(["TEST"], ["feature"], bar_count=1)["feature"].to_list())

    runtime.scheduler.register(before, at="session_open", phase=LifecyclePhase.PRE_OPEN)
    Engine(
        _instant_feed(prices_df=frame), Cash(), BacktestConfig(calendar="NYSE"), runtime=runtime
    ).run()
    assert seen == [[3.0]]


def test_previous_crypto_close_precedes_next_session_pre_open_at_the_same_instant():
    dates = [datetime(2025, 1, 2, 0, m, tzinfo=UTC) for m in [0, 1]]
    frame = pl.DataFrame({"timestamp": dates, "asset": ["TEST"] * 2, "close": [100.0, 101.0]})
    runtime = RuntimeServices(start=datetime(2025, 1, 1, 23, 59, tzinfo=UTC))
    trace = []

    def before(timestamp, data, context, broker):
        trace.append(
            (
                "before",
                timestamp,
                runtime.market.history(["TEST"], ["close"], bar_count=1)["close"].to_list(),
            )
        )

    def close(timestamp, data, context, broker):
        trace.append(("close", timestamp))

    runtime.scheduler.register(before, at="session_open", phase=LifecyclePhase.PRE_OPEN)
    runtime.scheduler.register(close, at="session_close", phase=LifecyclePhase.CLOSE)

    class Observe(Strategy):
        def on_data(self, timestamp, data, context, broker):
            trace.append(("data", timestamp))

    spec = FeedSpec(
        time_contract_version=2,
        calendar="crypto",
        data_frequency="1m",
        timestamp_semantics=TimestampSemantics.BAR_CLOSE,
        bar_label="right",
    )
    result = Engine(
        _instant_feed(prices_df=frame, feed_spec=spec), Observe(), BacktestConfig(), runtime=runtime
    ).run()
    assert trace == [
        ("data", dates[0]),
        ("close", dates[0]),
        ("before", dates[0], [100.0]),
        ("data", dates[1]),
    ]
    assert result.to_equity_dataframe()["timestamp"].to_list() == dates


def test_opening_callbacks_require_opening_instants_instead_of_completed_bar_inputs():
    runtime = RuntimeServices()
    runtime.scheduler.register(
        lambda *_args: None, at="session_open", phase=LifecyclePhase.OPENING_AUCTION
    )
    started = []

    class Observe(Cash):
        def on_start(self, broker):
            started.append(True)

    frame = pl.DataFrame(
        {"timestamp": [datetime(2025, 1, 2, 21, tzinfo=UTC)], "asset": ["TEST"], "close": [100.0]}
    )
    spec = FeedSpec(
        time_contract_version=2,
        timestamp_semantics=TimestampSemantics.BAR_CLOSE,
        bar_label="right",
        data_frequency="1d",
    )
    with pytest.raises(ValueError, match="opening.*event_time"):
        Engine(
            DataFeed(prices_df=frame, feed_spec=spec),
            Observe(),
            BacktestConfig(calendar="NYSE"),
            runtime=runtime,
        ).run()
    assert started == []
