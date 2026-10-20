"""Strategies must see only causally available observations, including non-trading warmup."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy
from ml4t.backtest.config import ExecutionMode


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
        DataFeed(prices_df=prices()),
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

    Engine(DataFeed(prices_df=frame), Read(), BacktestConfig(calendar=None), runtime=services).run()
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

    Engine(DataFeed(prices_df=frame), Read(), BacktestConfig(calendar=None), runtime=services).run()
    assert seen[1][0]["close"].to_list() == [None]
    assert seen[1][0]["volume"].to_list() == [None]
    assert seen[1][1]["close"].to_list() == [100.0]
    assert seen[1][1]["close_observed_at"].to_list() == [times[0]]
    assert frame.filter(pl.col("asset") == "TEST")["timestamp"].to_list() == [times[0], times[2]]
