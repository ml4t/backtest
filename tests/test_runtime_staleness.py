"""Read-view freshness must never manufacture executable quotes or volume."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import polars as pl
import pytest
from ml4t.specs import LifecyclePhase

from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy
from ml4t.backtest.config import ExecutionMode


def _sparse():
    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7]]
    return times, pl.DataFrame(
        {
            "timestamp": times,
            "asset": ["A", "B"],
            "close": [100.0, 50.0],
            "volume": [10.0, 20.0],
        }
    )


def test_per_field_fill_keeps_volume_missing_and_cannot_make_a_stale_order_execute():
    times, frame = _sparse()
    runtime = RuntimeServices()
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp != times[1]:
                return
            history = runtime.market.history(
                ["A"],
                ["price", "volume"],
                bar_count=2,
                forward_fill={"price": True, "volume": False},
            )
            assert history["price"].to_list() == [100.0, 100.0]
            assert history["volume"].to_list() == [10.0, None]
            assert history["price_observed_at"].to_list() == [times[0], times[0]]
            assert "volume_observed_at" not in history
            assert runtime.market.observation_age("A") == timedelta(days=1)
            assert runtime.market.is_stale("A")
            assert not runtime.market.is_stale("A", max_age=timedelta(days=1))
            assert not runtime.market.can_trade("A")
            seen.append(broker.submit_order("A", 1))

    result = Engine(
        DataFeed(prices_df=frame, feed_spec={"timestamp_semantics": "event_time"}),
        Read(),
        BacktestConfig(calendar=None, execution_mode=ExecutionMode.SAME_BAR),
        runtime=runtime,
    ).run()
    assert len(seen) == 1 and seen[0].filled_quantity == 0
    assert result.to_fills_dataframe().is_empty()


def test_expired_read_value_retains_the_real_observation_time_as_provenance():
    times, frame = _sparse()
    runtime = RuntimeServices()
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[1]:
                current = runtime.market.current(
                    ["A"], ["price"], forward_fill=True, max_age=timedelta(hours=23)
                )
                seen.append((current["price"][0], current["price_observed_at"][0]))
                assert runtime.market.observation_age("UNKNOWN") is None
                assert not runtime.market.is_stale("UNKNOWN")
                assert not runtime.market.can_trade("UNKNOWN")

    Engine(
        DataFeed(prices_df=frame, feed_spec={"timestamp_semantics": "event_time"}),
        Read(),
        BacktestConfig(calendar=None),
        runtime=runtime,
    ).run()
    assert seen == [(None, times[0])]


def test_preopen_explicit_stale_view_can_read_yesterdays_close_but_fills_only_at_actual_open():
    closed = datetime(2025, 1, 6, 21, tzinfo=UTC)
    opened = datetime(2025, 1, 7, 14, 30, tzinfo=UTC)
    runtime = RuntimeServices(start=datetime(2025, 1, 6, 20, tzinfo=UTC))
    seen = []

    def before(timestamp, data, context, broker):
        values = runtime.market.current(
            ["A"], ["price"], forward_fill=True, max_age=timedelta(days=1)
        )
        assert runtime.market.current(["A"], ["price"]).is_empty()
        assert not runtime.market.can_trade("A")
        seen.append((timestamp, values["price"][0], values["price_observed_at"][0]))
        broker.submit_order("A", 1)

    runtime.scheduler.register(
        before, phase=LifecyclePhase.PRE_OPEN, at="session_open", minutes_before=45
    )

    class Noop(Strategy):
        def on_data(self, timestamp, data, context, broker):
            pass

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {"timestamp": [closed, opened], "asset": ["A", "A"], "close": [100.0, 200.0]}
            ),
            feed_spec={"timestamp_semantics": "event_time"},
        ),
        Noop(),
        BacktestConfig(calendar="NYSE", execution_mode=ExecutionMode.NEXT_BAR),
        runtime=runtime,
    ).run()
    assert seen == [(datetime(2025, 1, 7, 13, 45, tzinfo=UTC), 100.0, closed)]
    assert result.to_fills_dataframe()["price"].to_list() == [200.0]
    assert result.to_fills_dataframe()["timestamp"].to_list() == [opened]


def test_one_nanosecond_age_is_not_rounded_to_a_fresh_observation():
    times = pd.to_datetime(["2025-01-06T21:00:00.000000001Z", "2025-01-06T21:00:00.000000002Z"])
    frame = pl.DataFrame(
        {
            "timestamp": pl.Series(times.asi8).cast(pl.Datetime("ns", "UTC")),
            "asset": ["A", "B"],
            "close": [100.0, 50.0],
        }
    )
    runtime = RuntimeServices()
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[1]:
                assert runtime.market.observation_age("A") == pd.Timedelta(1, unit="ns")
                assert runtime.market.is_stale("A")
                expired = runtime.market.current(
                    ["A"], ["price"], forward_fill=True, max_age=timedelta(0)
                )
                live = runtime.market.current(
                    ["A"], ["price"], forward_fill=True, max_age=pd.Timedelta(1, unit="ns")
                )
                seen.append((expired["price"][0], live["price"][0]))
                assert expired["price_observed_at"].cast(pl.Int64).to_list() == [times[0].value]

    Engine(
        DataFeed(prices_df=frame, feed_spec={"timestamp_semantics": "event_time"}),
        Read(),
        BacktestConfig(calendar=None),
        runtime=runtime,
    ).run()
    assert seen == [(None, 100.0)]


@pytest.mark.parametrize(
    "options",
    [
        {"forward_fill": {"price": True, "typo": False}},
        {"forward_fill": {"price": 1}},
        {"forward_fill": "yes"},
        {"max_age": timedelta(seconds=-1)},
        {"max_age": 1},
        {"forward_fill": {"volume": True}},
    ],
)
def test_invalid_fill_policies_fail_without_mutating_broker_state(options):
    times, frame = _sparse()
    runtime = RuntimeServices()

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            with pytest.raises(ValueError):
                runtime.market.current(["A"], ["price", "volume"], **options)
            assert broker.get_cash() == 100000 and broker.orders == []

    Engine(
        DataFeed(prices_df=frame, feed_spec={"timestamp_semantics": "event_time"}),
        Read(),
        BacktestConfig(calendar=None),
        runtime=runtime,
    ).run()
