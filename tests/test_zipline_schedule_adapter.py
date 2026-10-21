"""Scheduling must preserve calendar ordinals and callback order without early prices."""

import json
from datetime import datetime
from math import isnan
from pathlib import Path

import polars as pl
import pytest

from ml4t.backtest import DataFeed, RuntimeServices

_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/zipline_original/schedule-daily-reference.json").read_text()
)


def _feed():
    times = [datetime.fromisoformat(time) for name, time in _REFERENCE["trace"] if name == "handle"]
    return DataFeed(
        prices_df=pl.DataFrame(
            {
                "timestamp": times,
                "asset": ["A"] * 7,
                "close": [100.0 + i for i in range(7)],
                "volume": [1000] * 7,
            }
        ),
        feed_spec={
            "time_contract_version": 2,
            "calendar": "NYSE",
            "timestamp_semantics": "bar_close",
            "bar_label": "right",
            "bar_grid_anchor": "session_open",
            "partial_bar_policy": "include",
            "data_frequency": "1d",
        },
    )


def test_daily_rules_ignore_intraday_time_preserve_month_week_ordinals_and_halfday_policy():
    from ml4t.backtest.compat.zipline import run_algorithm
    from ml4t.backtest.compat.zipline.api import (
        date_rules,
        get_datetime,
        record,
        schedule_function,
        time_rules,
    )
    from ml4t.backtest.compat.zipline.assets import Asset

    trace = []

    def event(name):
        def callback(context, data):
            trace.append([name, get_datetime().isoformat()])
            record(last_schedule=name)

        return callback

    def initialize(context):
        schedule_function(
            event("every"), date_rules.every_day(), time_rules.market_open(minutes=30)
        )
        schedule_function(event("week_start"), date_rules.week_start(), time_rules.market_close())
        schedule_function(event("week_end"), date_rules.week_end(), time_rules.market_open())
        schedule_function(event("month_start_plus1"), date_rules.month_start(1))
        schedule_function(event("month_end"), date_rules.month_end())
        schedule_function(event("full_day"), date_rules.every_day(), half_days=False)

    def before(context, data):
        trace.append(["before", get_datetime().isoformat()])
        index = sum(name == "before" for name, _time in trace) - 1
        price = data.current(context.symbol("ALPHA"), "price")
        if index == 0:
            assert isnan(price)
        else:
            assert price == 100.0 + index - 1
        for call in [
            context.order,
            context.order_value,
            context.order_percent,
            context.order_target,
            context.order_target_value,
            context.order_target_percent,
        ]:
            with pytest.raises(RuntimeError, match="before_trading_start"):
                call(context.symbol("ALPHA"), 1)

    def handle(context, data):
        trace.append(["handle", get_datetime().isoformat()])

    runtime = RuntimeServices()
    performance = run_algorithm(
        feed=_feed(),
        assets=[Asset("A", "ALPHA")],
        initialize=initialize,
        before_trading_start=before,
        handle_data=handle,
        runtime=runtime,
    )
    # Published3.2 resets OncePerDay after24elapsed hours, incorrectly suppressing
    # the July3 early-close callback after July2's normal close. Native sessions
    # must still execute every_day on that valid observed session.
    corrected_event = ["every", "2025-07-03T17:00:00+00:00"]
    assert trace.count(corrected_event) == 1
    assert [event for event in trace if event != corrected_event] == _REFERENCE["trace"]
    assert performance.last_schedule.tolist() == _REFERENCE["last_schedule"]
    # A second run must not invoke callbacks bound to the earlier user context.
    run_algorithm(feed=_feed(), assets=[Asset("A", "ALPHA")], runtime=runtime)
    assert trace.count(corrected_event) == 1
    assert [event for event in trace if event != corrected_event] == _REFERENCE["trace"]


def test_minute_rules_match_actual_right_clock_offsets_and_early_close_endpoints():
    from ml4t.backtest.compat.zipline import run_algorithm
    from ml4t.backtest.compat.zipline.api import get_datetime, record, schedule_function, time_rules
    from ml4t.backtest.compat.zipline.assets import Asset

    reference = json.loads(
        (
            Path(__file__).parent / "fixtures/zipline_original/schedule-minute-reference.json"
        ).read_text()
    )
    times = [datetime.fromisoformat(t) for t in reference["minutes"]]
    feed = DataFeed(
        prices_df=pl.DataFrame(
            {
                "timestamp": times,
                "asset": ["A"] * len(times),
                "close": reference["close"],
                "volume": [1000] * len(times),
            }
        ),
        feed_spec={
            "time_contract_version": 2,
            "calendar": "NYSE",
            "data_frequency": "1m",
            "timestamp_semantics": "bar_close",
            "bar_label": "right",
            "bar_grid_anchor": "session_open",
            "partial_bar_policy": "include",
        },
    )
    trace = []

    def event(label):
        def callback(context, data):
            trace.append([label, get_datetime().isoformat()])
            record(last_schedule=label)

        return callback

    def initialize(context):
        schedule_function(event("default"))
        schedule_function(event("every_minute"), time_rule=time_rules.every_minute())
        schedule_function(event("open_default"), time_rule=time_rules.market_open())
        schedule_function(event("open30"), time_rule=time_rules.market_open(minutes=30))
        schedule_function(event("close_default"), time_rule=time_rules.market_close())
        schedule_function(event("close30"), time_rule=time_rules.market_close(minutes=30))
        schedule_function(event("full_day"), half_days=False)

    def handle(context, data):
        trace.append(["handle", get_datetime().isoformat()])

    performance = run_algorithm(
        feed=feed, assets=[Asset("A", "ALPHA")], initialize=initialize, handle_data=handle
    )
    assert trace == reference["trace"]
    assert (
        len(performance) == 600
    )  # Native account observations are minute events, not daily packets.
    assert performance.index.is_unique
    assert performance.last_schedule.iloc[-1] == "close_default"


def test_calendar_none_rejects_session_callbacks_without_adding_a_default_calendar_or_leaking_rules():
    from ml4t.backtest.compat.zipline import run_algorithm
    from ml4t.backtest.compat.zipline.api import schedule_function
    from ml4t.backtest.compat.zipline.assets import Asset

    runtime = RuntimeServices()
    feed = _feed()

    def initialize(context):
        schedule_function(lambda _context, _data: None)

    without_calendar = feed.feed_spec.with_overrides(calendar=None)
    with pytest.raises(ValueError, match="calendar=None disables"):
        run_algorithm(
            feed=DataFeed(prices_df=feed.prices, feed_spec=without_calendar),
            assets=[Asset("A", "ALPHA")],
            initialize=initialize,
            runtime=runtime,
        )
    performance = run_algorithm(
        feed=DataFeed(prices_df=feed.prices, feed_spec=without_calendar),
        assets=[Asset("A", "ALPHA")],
        runtime=runtime,
    )
    assert performance.attrs["backtest_result"].config.resolved_calendar is None


def test_missing_minute_endpoint_does_not_fabricate_a_price_event_to_trigger_callback():
    from datetime import UTC

    from ml4t.backtest.compat.zipline import run_algorithm
    from ml4t.backtest.compat.zipline.api import get_datetime, schedule_function, time_rules
    from ml4t.backtest.compat.zipline.assets import Asset

    times = [datetime(2025, 7, 2, 13, m, tzinfo=UTC) for m in [31, 33]]
    feed = DataFeed(
        prices_df=pl.DataFrame(
            {"timestamp": times, "asset": ["A"] * 2, "close": [100.0, 102.0], "volume": [1000] * 2}
        ),
        feed_spec={
            "time_contract_version": 2,
            "calendar": "NYSE",
            "data_frequency": "1m",
            "timestamp_semantics": "bar_close",
            "bar_label": "right",
            "bar_grid_anchor": "session_open",
            "partial_bar_policy": "include",
        },
    )
    callbacks = []

    def initialize(context):
        schedule_function(
            lambda _context, _data: callbacks.append(("missing", get_datetime())),
            time_rule=time_rules.market_open(minutes=2),
        )
        with pytest.warns(UserWarning, match="standalone time rule"):
            schedule_function(
                lambda _context, _data: callbacks.append(("present", get_datetime())),
                time_rules.market_open(minutes=3),
            )

    performance = run_algorithm(feed=feed, assets=[Asset("A", "ALPHA")], initialize=initialize)
    assert callbacks == [("present", times[1])]
    assert performance.index.to_pydatetime().tolist() == times


def test_failed_pre_callback_clears_its_api_scope_and_registration_before_runtime_reuse():
    from ml4t.backtest.compat.zipline import run_algorithm
    from ml4t.backtest.compat.zipline.api import get_datetime
    from ml4t.backtest.compat.zipline.assets import Asset

    runtime = RuntimeServices()

    def before(context, data):
        raise ValueError("user PRE failure")

    with pytest.raises(ValueError, match="user PRE failure"):
        run_algorithm(
            feed=_feed(), assets=[Asset("A", "ALPHA")], before_trading_start=before, runtime=runtime
        )
    with pytest.raises(RuntimeError, match="active migration callback"):
        get_datetime()
    performance = run_algorithm(feed=_feed(), assets=[Asset("A", "ALPHA")], runtime=runtime)
    assert len(performance) == 7
