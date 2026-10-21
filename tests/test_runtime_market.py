"""Strategies must see only causally available observations, including non-trading warmup."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from ml4t.specs import FeedSpec, TimestampSemantics

from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy
from ml4t.backtest.config import ExecutionMode


def _instant_feed(**kwargs):
    spec = FeedSpec.from_any(kwargs.pop("feed_spec", None))
    if spec.timestamp_semantics is None:
        spec = spec.with_overrides(timestamp_semantics=TimestampSemantics.EVENT_TIME)
    return DataFeed(feed_spec=spec, **kwargs)


def prices(extra_future=False):
    times = [datetime(2025, 1, day, 16, tzinfo=UTC) for day in [2, 3, 6, 7]]
    if extra_future:
        times.append(datetime(2025, 1, 8, 16, tzinfo=UTC))
    return pl.DataFrame(
        {
            "timestamp": times,
            "asset": ["TEST"] * len(times),
            "close": [100.0, 110.0, 120.0, 130.0, 99999.0][: len(times)],
        }
    )


def test_warmup_history_is_visible_but_cannot_create_prestart_trades_or_accounts():
    start = datetime(2025, 1, 6, 16, tzinfo=UTC)
    services = RuntimeServices(start=start)
    seen = []

    class Average(Strategy):
        def on_data(self, timestamp, data, context, broker):
            history = services.market.history(["TEST"], ["close"], bar_count=3)
            seen.append((timestamp, history["close"].to_list(), dict(context)))
            if history["close"][-1] > history["close"].mean() and not broker.positions:
                broker.submit_order("TEST", 10)

    result = Engine(
        _instant_feed(prices_df=prices()),
        Average(),
        BacktestConfig(initial_cash=10000, calendar=None, execution_mode=ExecutionMode.SAME_BAR),
        runtime=services,
    ).run()
    assert seen == [
        (start, [100.0, 110.0, 120.0], {}),
        (datetime(2025, 1, 7, 16, tzinfo=UTC), [110.0, 120.0, 130.0], {}),
    ]
    assert result.to_fills_dataframe()["timestamp"].to_list() == [start]
    assert result.to_fills_dataframe()["price"].to_list() == [120.0]
    assert result.to_equity_dataframe()["timestamp"].to_list() == [row[0] for row in seen]
    assert result.to_equity_dataframe()["equity"].to_list() == [10000.0, 10100.0]
    with pytest.raises(RuntimeError, match="active run"):
        services.market.history(["TEST"], ["close"], bar_count=1)


@pytest.mark.parametrize("entry_point", ["from_config", "run_backtest"])
def test_convenience_entry_points_bind_runtime_warmup_history_and_release_it_after_run(entry_point):
    from ml4t.backtest import run_backtest

    start = datetime(2025, 1, 6, 16, tzinfo=UTC)
    services = RuntimeServices(start=start)
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            seen.append(
                services.market.history(["TEST"], ["close"], bar_count=3)["close"].to_list()
            )
            if timestamp == start:
                broker.submit_order("TEST", 1)

    config = BacktestConfig(calendar=None, execution_mode=ExecutionMode.SAME_BAR)
    if entry_point == "from_config":
        result = Engine.from_config(
            _instant_feed(prices_df=prices()), Read(), config, runtime=services
        ).run()
    else:
        result = run_backtest(
            prices(),
            Read(),
            config=config,
            feed_spec=FeedSpec(timestamp_semantics=TimestampSemantics.EVENT_TIME),
            runtime=services,
        )
    assert seen == [[100.0, 110.0, 120.0], [110.0, 120.0, 130.0]]
    assert [(fill.timestamp, fill.price, fill.quantity) for fill in result.fills] == [
        (start, 120.0, 1)
    ]
    assert result.to_equity_dataframe()["timestamp"].to_list() == [
        start,
        datetime(2025, 1, 7, 16, tzinfo=UTC),
    ]
    with pytest.raises(RuntimeError, match="active run"):
        services.market.history(["TEST"], ["close"], bar_count=1)


@pytest.mark.parametrize("extra_future", [False, True])
def test_future_prices_and_late_known_features_cannot_change_an_earlier_decision(extra_future):
    frame = prices(extra_future).with_columns(
        pl.when(pl.col("timestamp").dt.day() == 3)
        .then(pl.lit(datetime(2025, 1, 7, 16, tzinfo=UTC)))
        .otherwise(pl.col("timestamp"))
        .alias("close_known_at")
    )
    services = RuntimeServices(
        end=datetime(2025, 1, 6, 16, tzinfo=UTC), known_at_cols={"close": "close_known_at"}
    )
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            seen.append(
                services.market.history(["TEST"], ["close"], bar_count=3)["close"].to_list()
            )

    Engine(
        _instant_feed(prices_df=frame), Read(), BacktestConfig(calendar=None), runtime=services
    ).run()
    assert seen == [[100.0], [100.0, None], [100.0, None, 120.0]]


def test_stale_history_is_an_explicit_view_and_missing_volume_stays_unknown():
    times = [datetime(2025, 1, 2, 15, tzinfo=UTC) + timedelta(minutes=i) for i in range(3)]
    frame = pl.DataFrame(
        {
            "timestamp": [times[0], times[1], times[2]],
            "asset": ["TEST", "OTHER", "TEST"],
            "close": [100.0, 1.0, 110.0],
        }
    )
    services = RuntimeServices()
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            seen.append(
                (
                    services.market.current(["TEST"], ["close", "volume"]),
                    services.market.current(["TEST"], ["close"], forward_fill=True),
                )
            )

    Engine(
        _instant_feed(prices_df=frame), Read(), BacktestConfig(calendar=None), runtime=services
    ).run()
    assert seen[1][0]["close"].to_list() == [None]
    assert seen[1][0]["volume"].to_list() == [None]
    assert seen[1][1]["close"].to_list() == [100.0]
    assert seen[1][1]["close_observed_at"].to_list() == [times[0]]
    assert frame.filter(pl.col("asset") == "TEST")["timestamp"].to_list() == [times[0], times[2]]


def test_late_price_cannot_reach_strategy_or_execute_until_a_known_quote():
    frame = prices().with_columns(
        pl.when(pl.col("timestamp").dt.day() == 3)
        .then(pl.lit(datetime(2025, 1, 7, 16, tzinfo=UTC)))
        .otherwise(pl.col("timestamp"))
        .alias("known")
    )
    runtime = RuntimeServices(known_at_cols={"close": "known"})

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 3:
                assert data["TEST"]["close"] is None
                assert broker.get_mark_price("TEST") is None
                broker.submit_order("TEST", 1)

    result = Engine(
        _instant_feed(prices_df=frame),
        Buy(),
        BacktestConfig(calendar=None, execution_mode=ExecutionMode.SAME_BAR),
        runtime=runtime,
    ).run()
    assert result.to_fills_dataframe()["timestamp"].to_list() == [
        datetime(2025, 1, 6, 16, tzinfo=UTC)
    ]
    assert result.to_fills_dataframe()["price"].to_list() == [120.0]
    assert frame["close"].to_list() == [100.0, 110.0, 120.0, 130.0]


def test_failed_nested_binding_does_not_detach_the_active_outer_market():
    runtime = RuntimeServices()

    class Outer(Strategy):
        def on_data(self, timestamp, data, context, broker):
            with pytest.raises(RuntimeError, match="active run"):
                Engine(
                    _instant_feed(prices_df=prices()),
                    self,
                    BacktestConfig(calendar=None),
                    runtime=runtime,
                ).run()
            assert runtime.market.current(["TEST"], ["close"])["close"].to_list() == [
                data["TEST"]["close"]
            ]
            raise ValueError("abort outer callback")

    with pytest.raises(ValueError, match="abort outer"):
        Engine(
            _instant_feed(prices_df=prices()),
            Outer(),
            BacktestConfig(calendar=None),
            runtime=runtime,
        ).run()
    with pytest.raises(RuntimeError, match="active run"):
        runtime.market.current(["TEST"], ["close"])


def test_last_trade_requires_observed_volume_and_stale_quotes_are_not_executable():
    times = [datetime(2025, 1, 2, 15, tzinfo=UTC) + timedelta(minutes=i) for i in range(3)]
    frame = pl.DataFrame(
        {
            "timestamp": times,
            "asset": ["TEST"] * 3,
            "close": [100.0, 101.0, 102.0],
            "volume": [10.0, 0.0, 20.0],
        }
    )
    runtime = RuntimeServices()
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            seen.append(
                (
                    runtime.market.last_traded("TEST"),
                    runtime.market.last_observed("TEST"),
                    runtime.market.can_trade("TEST", require_volume=True),
                )
            )

    Engine(
        _instant_feed(prices_df=frame), Read(), BacktestConfig(calendar=None), runtime=runtime
    ).run()
    assert seen == [
        (times[0], times[0], True),
        (times[0], times[1], False),
        (times[2], times[2], True),
    ]


def test_market_view_retained_from_previous_run_cannot_read_during_a_new_run():
    runtime = RuntimeServices()
    views = []

    class Retain(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if views:
                with pytest.raises(RuntimeError, match="active run"):
                    views[0].history(["TEST"], ["close"], bar_count=1)
            if not views:
                views.append(runtime.market)

    single = prices().head(1)
    Engine(
        _instant_feed(prices_df=single), Retain(), BacktestConfig(calendar=None), runtime=runtime
    ).run()
    Engine(
        _instant_feed(prices_df=single), Retain(), BacktestConfig(calendar=None), runtime=runtime
    ).run()


def test_initialization_uses_declared_start_not_the_future_first_trading_bar():
    dates = [datetime(2025, 1, 3, 21, tzinfo=UTC), datetime(2025, 1, 6, 21, tzinfo=UTC)]
    start = datetime(2025, 1, 6, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "timestamp": dates,
            "asset": ["TEST"] * 2,
            "close": [100.0, 101.0],
            "feature": [3.0, 9.0],
            "known": [datetime(2025, 1, 6, 20, tzinfo=UTC), dates[1]],
        }
    )
    runtime = RuntimeServices(start=start, known_at_cols={"feature": "known"})
    seen = []

    class Read(Strategy):
        def on_start(self, broker):
            seen.append(
                runtime.market.history(["TEST"], ["feature"], bar_count=1)["feature"].to_list()
            )

        def on_data(self, timestamp, data, context, broker):
            seen.append(
                runtime.market.history(["TEST"], ["feature"], bar_count=2)["feature"].to_list()
            )

    Engine(
        _instant_feed(prices_df=frame), Read(), BacktestConfig(calendar=None), runtime=runtime
    ).run()
    assert seen == [[None], [3.0, 9.0]]


def test_unpublished_quote_depth_cannot_leak_from_the_history_view_into_executable_data():
    from ml4t.backtest.config import ExecutionPrice
    from ml4t.backtest.execution import VolumeParticipationLimit

    dates = [datetime(2025, 1, 2, 15, tzinfo=UTC), datetime(2025, 1, 2, 15, 1, tzinfo=UTC)]
    frame = pl.DataFrame(
        {
            "timestamp": dates,
            "asset": ["TEST"] * 2,
            "close": [100.0] * 2,
            "ask": [100.0] * 2,
            "ask_size": [1000.0] * 2,
            "known": [dates[1]] * 2,
        }
    )
    runtime = RuntimeServices(known_at_cols={"ask_size": "known"})
    seen = []

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            observed = runtime.market.current(["TEST"], ["ask_size"])["ask_size"].to_list()[0]
            seen.append((data["TEST"].get("ask_size"), observed))
            if timestamp == dates[0]:
                broker.submit_order("TEST", 10)

    result = Engine(
        _instant_feed(prices_df=frame, ask_col="ask", ask_size_col="ask_size"),
        Buy(),
        BacktestConfig(
            initial_cash=10000,
            calendar=None,
            execution_mode=ExecutionMode.SAME_BAR,
            execution_price=ExecutionPrice.QUOTE_SIDE,
        ),
        runtime=runtime,
        execution_limits=VolumeParticipationLimit(max_participation=0.1),
    ).run()
    assert result.to_fills_dataframe()["timestamp"].to_list() == [dates[1]]
    assert result.to_fills_dataframe()["quantity"].to_list() == [10.0]
    assert seen == [(None, None), (1000.0, 1000.0)]


@pytest.mark.parametrize("role", [None, TimestampSemantics.SESSION_LABEL])
def test_physical_runtime_cutoffs_reject_undeclared_or_session_label_clocks(role):
    dates = [datetime(2025, 1, d, tzinfo=UTC) for d in [6, 7]]
    frame = pl.DataFrame({"timestamp": dates, "asset": ["TEST"] * 2, "close": [999.0, 100.0]})
    seen = []
    runtime = RuntimeServices(start=datetime(2025, 1, 6, 13, 45, tzinfo=UTC))

    class Read(Strategy):
        def on_start(self, broker):
            seen.append(runtime.market.history(["TEST"], ["close"], bar_count=1)["close"].to_list())

        def on_data(self, *args):
            pass

    with pytest.raises(ValueError, match="timestamp_semantics"):
        Engine(
            DataFeed(prices_df=frame, feed_spec=FeedSpec(timestamp_semantics=role)),
            Read(),
            BacktestConfig(calendar=None),
            runtime=runtime,
        ).run()
    assert seen == []
