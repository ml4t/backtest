"""Execution windows and before-event reads must not publish or fill a future bar."""

from datetime import UTC, datetime

import polars as pl
import pytest
from ml4t.specs import LifecyclePhase

from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy
from ml4t.backtest.execution import RebalanceSchedule

TIMES = [datetime(2025, 1, day, 21, tzinfo=UTC) for day in [6, 7, 8, 9]]


def test_windows_keep_warmup_history_and_before_event_callbacks_precede_quote_updates_and_fills():
    prices = pl.DataFrame(
        {"timestamp": TIMES, "asset": ["A"] * 4, "close": [100.0, 101.0, 102.0, 103.0]}
    )
    runtime = RuntimeServices(execution_windows=[(TIMES[1], TIMES[2])])
    before, after = [], []

    def observe(timestamp, data, context, broker):
        assert data == {} and context == {}
        assert broker.get_account_snapshot().timestamp == timestamp
        value = runtime.market.current(["A"], ["close"], forward_fill=True)["close"][0]
        before.append((timestamp, value, broker.get_position("A")))

    runtime.scheduler.register(
        observe,
        phase=LifecyclePhase.INTRABAR,
        at="clock",
        schedule=RebalanceSchedule.explicit_timestamps(TIMES[1:3]),
    )

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            history = runtime.market.history(["A"], ["close"], bar_count=4)
            after.append((timestamp, history["close"].to_list()))
            broker.submit_order("A", 1)

    result = Engine(
        DataFeed(
            prices_df=prices,
            feed_spec={
                "time_contract_version": 2,
                "calendar": None,
                "timestamp_semantics": "bar_close",
            },
        ),
        Buy(),
        BacktestConfig(calendar=None),
        runtime=runtime,
    ).run()
    assert [(time, value, position is None) for time, value, position in before] == [
        (TIMES[1], 100.0, True),
        (TIMES[2], 101.0, True),
    ]
    assert after == [(TIMES[1], [100.0, 101.0]), (TIMES[2], [100.0, 101.0, 102.0])]
    assert result.to_fills_dataframe()["price"].to_list() == [102.0]
    assert result.to_fills_dataframe()["quantity"].to_list() == [1.0]
    assert result.to_equity_dataframe()["timestamp"].to_list() == TIMES[1:3]
    assert runtime.execution_windows == ((TIMES[1], TIMES[2]),)


@pytest.mark.parametrize(
    "windows",
    [
        [(TIMES[2], TIMES[1])],
        [(TIMES[0], TIMES[2]), (TIMES[1], TIMES[3])],
        [(TIMES[1], TIMES[2]), (TIMES[0], TIMES[0])],
        [(TIMES[0], "tomorrow")],
    ],
)
def test_invalid_execution_windows_fail_before_any_run(windows):
    with pytest.raises((TypeError, ValueError), match="window"):
        RuntimeServices(execution_windows=windows)


@pytest.mark.parametrize("offset", ["1h", "1ns"])
def test_clock_without_an_observation_does_not_fill_pending_orders_or_create_an_account_sample(
    offset,
):
    import pandas as pd

    prices = pl.DataFrame(
        {"timestamp": TIMES, "asset": ["A"] * 4, "close": [100.0, 101.0, 102.0, 103.0]}
    )
    runtime = RuntimeServices(start=TIMES[1], end=TIMES[2])
    scheduled = pd.Timestamp(TIMES[2]) - pd.Timedelta(offset)
    seen = []

    def observe(timestamp, data, context, broker):
        assert timestamp == scheduled
        assert broker.get_account_snapshot().timestamp == scheduled
        assert broker.get_position("A") is None
        assert runtime.market.current(["A"], ["close"]).is_empty()
        assert runtime.market.current(["A"], ["close"], forward_fill=True)["close"][0] == 101.0
        seen.append(timestamp)

    runtime.scheduler.register(
        observe,
        phase=LifecyclePhase.INTRABAR,
        at="clock",
        schedule=RebalanceSchedule.explicit_timestamps([scheduled]),
    )

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[1]:
                broker.submit_order("A", 1)

    result = Engine(
        DataFeed(
            prices_df=prices,
            feed_spec={
                "time_contract_version": 2,
                "calendar": None,
                "timestamp_semantics": "bar_close",
            },
        ),
        Buy(),
        BacktestConfig(calendar=None),
        runtime=runtime,
    ).run()
    assert seen == [scheduled]
    assert result.to_equity_dataframe()["timestamp"].to_list() == TIMES[1:3]
    assert result.to_fills_dataframe()["price"].to_list() == [102.0]


@pytest.mark.parametrize("action", ["order", "rebalance"])
def test_clock_order_cannot_immediately_fill_against_the_previous_observation(action):
    from ml4t.backtest.config import ExecutionMode
    from ml4t.backtest.types import OrderStatus

    prices = pl.DataFrame(
        {"timestamp": TIMES, "asset": ["A"] * 4, "close": [100.0, 101.0, 102.0, 103.0]}
    )
    runtime = RuntimeServices(start=TIMES[1], end=TIMES[2])
    scheduled = TIMES[2].replace(hour=20)

    def submit(timestamp, data, context, broker):
        if action == "order":
            order = broker.submit_order("A", 1)
            assert order.status is OrderStatus.PENDING
        else:
            broker.rebalance_to_weights({"A": 0.01})
        assert broker.get_position("A") is None

    runtime.scheduler.register(
        submit,
        phase=LifecyclePhase.INTRABAR,
        at="clock",
        schedule=RebalanceSchedule.explicit_timestamps([scheduled]),
    )

    class Read(Strategy):
        def on_data(self, *args):
            pass

    result = Engine(
        DataFeed(
            prices_df=prices,
            feed_spec={
                "time_contract_version": 2,
                "calendar": None,
                "timestamp_semantics": "bar_close",
            },
        ),
        Read(),
        BacktestConfig(calendar=None, execution_mode=ExecutionMode.SAME_BAR, immediate_fill=True),
        runtime=runtime,
    ).run()
    assert result.to_fills_dataframe()["price"].to_list() == [102.0]
    assert result.to_fills_dataframe()["timestamp"].to_list() == [TIMES[2]]


def test_explicit_run_end_excludes_later_clock_registrations_like_later_market_observations():
    prices = pl.DataFrame(
        {"timestamp": TIMES, "asset": ["A"] * 4, "close": [100.0, 101.0, 102.0, 103.0]}
    )
    runtime = RuntimeServices(end=TIMES[2])
    seen = []
    runtime.scheduler.register(
        lambda *args: seen.append(args),
        phase=LifecyclePhase.INTRABAR,
        at="clock",
        schedule=RebalanceSchedule.explicit_timestamps([TIMES[3]]),
    )

    class Read(Strategy):
        def on_data(self, *args):
            pass

    result = Engine(
        DataFeed(
            prices_df=prices,
            feed_spec={
                "time_contract_version": 2,
                "calendar": None,
                "timestamp_semantics": "bar_close",
            },
        ),
        Read(),
        BacktestConfig(calendar=None),
        runtime=runtime,
    ).run()
    assert seen == []
    assert result.to_equity_dataframe()["timestamp"].to_list() == TIMES[:3]
