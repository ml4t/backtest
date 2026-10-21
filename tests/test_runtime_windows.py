"""Window declarations must select time, not silently substitute observed row counts."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from ml4t.specs import FeedSpec, TimestampSemantics, WindowUnit

from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy


def _run(frame, spec, query, *, start=None, end=None, calendar="NYSE"):
    runtime = RuntimeServices(start=start, end=end)
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            seen.append((timestamp, query(runtime.market)))

    result = Engine(
        DataFeed(prices_df=frame, feed_spec=spec),
        Read(),
        BacktestConfig(calendar=calendar),
        runtime=runtime,
    ).run()
    return seen, result


def _frame(times, values=None):
    return pl.DataFrame(
        {
            "timestamp": times,
            "asset": ["TEST"] * len(times),
            "close": values or [float(i + 1) for i in range(len(times))],
        }
    )


def _grid(**kwargs):
    return replace(
        FeedSpec(
            time_contract_version=2,
            timezone="UTC",
            data_frequency="5m",
            timestamp_semantics=TimestampSemantics.BAR_CLOSE,
            bar_grid_anchor="session_open",
            bar_label="right",
            partial_bar_policy="include",
        ),
        **kwargs,
    )


def test_expected_bar_history_preserves_a_missing_bar_instead_of_reaching_back_an_extra_row():
    times = [datetime(2025, 1, 2, 14, m, tzinfo=UTC) for m in [35, 45]]
    seen, result = _run(
        _frame(times, [100.0, 110.0]),
        _grid(),
        lambda market: (
            market.history(["TEST"], ["close"], bar_count=2, unit=WindowUnit.EXPECTED_BAR),
            market.history(["TEST"], ["close"], bar_count=2),
            market.history(
                ["TEST"], ["close"], bar_count=2, unit="expected_bar", forward_fill=True
            ),
        ),
    )
    expected, observed, stale = seen[-1][1]
    assert expected["timestamp"].to_list() == [datetime(2025, 1, 2, 14, 40, tzinfo=UTC), times[-1]]
    assert expected["close"].to_list() == [None, 110.0]
    assert observed["close"].to_list() == [100.0, 110.0]
    assert stale["close"].to_list() == [100.0, 110.0]
    assert stale["close_observed_at"].to_list() == times
    assert result.to_equity_dataframe()["timestamp"].to_list() == times


@pytest.mark.parametrize(
    "partial,expected",
    [
        (
            "include",
            [datetime(2025, 7, 3, 16, 30, tzinfo=UTC), datetime(2025, 7, 3, 17, tzinfo=UTC)],
        ),
        (
            "exclude",
            [datetime(2025, 7, 3, 15, 30, tzinfo=UTC), datetime(2025, 7, 3, 16, 30, tzinfo=UTC)],
        ),
    ],
)
def test_expected_grid_respects_the_early_close_and_partial_bar_policy(partial, expected):
    now = datetime(2025, 7, 3, 17, tzinfo=UTC)
    spec = _grid(data_frequency="1h", partial_bar_policy=partial)
    frame = _frame(expected) if partial == "include" else _frame(expected + [now])
    # An auxiliary clock tick is not needed: on_start can query the elapsed cutoff.
    runtime = RuntimeServices(start=now)
    seen = []

    class Read(Strategy):
        def on_start(self, broker):
            seen.append(
                runtime.market.history(["TEST"], ["close"], bar_count=2, unit="expected_bar")
            )

        def on_data(self, *args):
            pass

    Engine(
        DataFeed(prices_df=frame, feed_spec=spec), Read(), BacktestConfig(), runtime=runtime
    ).run()
    # Initialization is exclusive of the event at start, including a partial close.
    before = expected[:-1] if partial == "include" else expected
    assert seen[0]["timestamp"].to_list()[-len(before) :] == before


def test_expected_grid_restarts_after_lunch_without_inventing_a_break_bar():
    times = [datetime(2025, 1, 2, h, m, tzinfo=UTC) for h, m in [(3, 30), (4, 0), (6, 0)]]
    seen, _ = _run(
        _frame(times),
        _grid(data_frequency="1h"),
        lambda m: m.history(["TEST"], ["close"], bar_count=3, unit="expected_bar"),
        calendar="XHKG",
    )
    assert seen[-1][1]["timestamp"].to_list() == times
    assert seen[-1][1]["close"].to_list() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("append_future", [False, True])
def test_daily_expected_windows_follow_dst_and_ignore_appended_future_prices(append_future):
    times = [datetime(2025, 3, 7, 21, tzinfo=UTC), datetime(2025, 3, 10, 20, tzinfo=UTC)]
    if append_future:
        times.append(datetime(2025, 3, 11, 20, tzinfo=UTC))
    seen, result = _run(
        _frame(times),
        _grid(data_frequency="1d"),
        lambda m: m.history(["TEST"], ["close"], bar_count=2, unit="expected_bar"),
        end=datetime(2025, 3, 10, 20, tzinfo=UTC),
    )
    assert seen[-1][1]["timestamp"].to_list() == times[:2]
    assert seen[-1][1]["close"].to_list() == [1.0, 2.0]
    assert result.to_equity_dataframe()["timestamp"].to_list() == times[:2]


@pytest.mark.parametrize("partial", ["include", "exclude"])
def test_left_labeled_grid_obeys_partial_bar_policy(partial):
    # 7-minute bars restart at 14:30. The regular close at 21:00 leaves five minutes.
    times = [datetime(2025, 1, 2, 20, m, tzinfo=UTC) for m in [34, 41, 48, 55]]
    # The final left-labeled partial bar starts at 20:55; no 21:00 price is invented.
    spec = _grid(
        data_frequency="7m",
        timestamp_semantics="event_time",
        bar_label="left",
        partial_bar_policy=partial,
    )
    frame = _frame(times if partial == "include" else times[:-1])
    seen, _ = _run(
        frame, spec, lambda m: m.history(["TEST"], ["close"], bar_count=2, unit="expected_bar")
    )
    assert seen[-1][1]["timestamp"].to_list() == (
        times[-2:] if partial == "include" else times[-3:-1]
    )


def test_exclusive_initialization_after_a_partial_close_still_includes_that_closed_bar():
    start = datetime(2025, 7, 3, 17, 30, tzinfo=UTC)
    times = [
        datetime(2025, 7, 3, 16, 30, tzinfo=UTC),
        datetime(2025, 7, 3, 17, tzinfo=UTC),
        datetime(2025, 7, 7, 14, 30, tzinfo=UTC),
    ]
    runtime = RuntimeServices(start=start)
    seen = []

    class Read(Strategy):
        def on_start(self, broker):
            seen.append(
                runtime.market.history(["TEST"], ["close"], bar_count=2, unit="expected_bar")
            )

        def on_data(self, *args):
            pass

    Engine(
        DataFeed(prices_df=_frame(times), feed_spec=_grid(data_frequency="1h")),
        Read(),
        BacktestConfig(),
        runtime=runtime,
    ).run()
    assert seen[0]["timestamp"].to_list() == times[:2]
    assert seen[0]["close"].to_list() == [1.0, 2.0]


def test_session_window_uses_overnight_exchange_sessions_and_retains_original_instants():
    times = [
        datetime(2025, 7, 3, 15, tzinfo=UTC),
        datetime(2025, 7, 6, 23, 30, tzinfo=UTC),
        datetime(2025, 7, 7, 0, 30, tzinfo=UTC),
    ]
    spec = FeedSpec(timestamp_semantics="event_time")
    seen, _ = _run(
        _frame(times),
        spec,
        lambda m: m.history(["TEST"], ["close"], bar_count=1, unit="session"),
        calendar="CME_Equity",
    )
    assert seen[-1][1]["timestamp"].to_list() == times[1:]
    assert seen[-1][1]["close"].to_list() == [2.0, 3.0]


def test_elapsed_window_is_left_open_and_does_not_turn_a_weekend_into_sessions():
    times = [
        datetime(2025, 1, 3, 21, tzinfo=UTC),
        datetime(2025, 1, 6, 20, tzinfo=UTC),
        datetime(2025, 1, 6, 21, tzinfo=UTC),
    ]
    seen, _ = _run(
        _frame(times),
        FeedSpec(timestamp_semantics="event_time"),
        lambda m: m.history(["TEST"], ["close"], unit="elapsed", duration=timedelta(hours=1)),
        calendar=None,
    )
    assert seen[-1][1]["timestamp"].to_list() == [times[-1]]


@pytest.mark.parametrize("unit", ["session", "expected_bar"])
def test_calendar_free_history_never_substitutes_a_us_session_or_continuous_grid(unit):
    frame = _frame([datetime(2025, 1, 2, 14, 35, tzinfo=UTC)])
    with pytest.raises(ValueError, match="calendar"):
        _run(
            frame,
            _grid(),
            lambda m: m.history(["TEST"], ["close"], bar_count=1, unit=unit),
            calendar=None,
        )


def test_expected_bar_history_rejects_an_observed_off_grid_price_instead_of_hiding_it():
    times = [datetime(2025, 1, 2, 14, m, tzinfo=UTC) for m in [36, 45]]
    with pytest.raises(ValueError, match="off-grid"):
        _run(
            _frame(times),
            _grid(),
            lambda m: m.history(["TEST"], ["close"], bar_count=3, unit="expected_bar"),
        )


@pytest.mark.parametrize(
    "overrides", [{"bar_grid_anchor": None}, {"partial_bar_policy": None}, {"bar_label": None}]
)
def test_expected_bar_history_requires_complete_grid_declarations(overrides):
    frame = _frame([datetime(2025, 1, 2, 14, 35, tzinfo=UTC)])
    with pytest.raises(ValueError, match="grid"):
        _run(
            frame,
            _grid(**overrides),
            lambda m: m.history(["TEST"], ["close"], bar_count=1, unit="expected_bar"),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unit": "elapsed", "duration": timedelta(0)},
        {"unit": "elapsed", "duration": timedelta(minutes=1), "bar_count": 1},
        {"unit": "observation", "bar_count": 1, "duration": timedelta(minutes=1)},
        {"unit": "session", "bar_count": True},
    ],
)
def test_window_requests_reject_ambiguous_counts_and_durations(kwargs):
    frame = _frame([datetime(2025, 1, 2, 14, 35, tzinfo=UTC)])
    with pytest.raises(ValueError):
        _run(frame, _grid(), lambda m: m.history(["TEST"], ["close"], **kwargs))
