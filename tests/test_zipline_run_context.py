"""Session labels must bound actual callbacks without discarding causal warmup."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from ml4t.backtest import DataFeed, RuntimeServices
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset

_ROOT = Path(__file__).parent / "fixtures/zipline_original"
_REFERENCE = json.loads((_ROOT / "environment-reference.json").read_text())


def _feed():
    clocks = json.loads((_ROOT / "min-leverage-reference.json").read_text())["minutes"]
    times = [datetime.fromisoformat(value) for value in clocks]
    return DataFeed(
        prices_df=pl.DataFrame(
            {
                "timestamp": times,
                "asset": ["A"] * len(times),
                "close": [100.0] * len(times),
                "volume": [1000.0] * len(times),
            }
        ),
        feed_spec={
            "calendar": "NYSE",
            "time_contract_version": 2,
            "timestamp_semantics": "bar_close",
            "bar_label": "right",
            "data_frequency": "1m",
            "bar_grid_anchor": "session_open",
            "partial_bar_policy": "include",
        },
    )


def test_session_bounds_and_environment_match_published_weekend_end_and_prestart_warmup():
    observed, environments, histories = [], [], []

    def initialize(context):
        context.a = context.symbol("TEST")
        environment = context.get_environment("*")
        environments.append(
            {
                key: value.isoformat() if hasattr(value, "isoformat") else value
                for key, value in environment.items()
            }
        )
        assert context.get_environment() == "ml4t.backtest"
        environment["capital_base"] = -1
        assert context.get_environment("capital_base") == 1000
        with pytest.raises(ValueError, match="environment"):
            context.get_environment("not_a_field")

    def handle(context, data):
        observed.append(context.get_datetime().isoformat())
        if len(observed) == 1:
            series = data.history(context.a, "price", 2, "1m")
            histories.extend(
                {"timestamp": stamp.isoformat(), "value": float(value)}
                for stamp, value in series.items()
            )

    result = run_algorithm(
        feed=_feed(),
        assets=[Asset("A", "TEST")],
        initialize=initialize,
        handle_data=handle,
        start="2025-07-03",
        end="2025-07-05",
        capital_base=1000,
    )
    expected = dict(_REFERENCE["environment"], platform="ml4t.backtest")
    assert environments == [expected]
    assert len(observed) == _REFERENCE["callback_count"]
    assert observed[0] == _REFERENCE["first"] and observed[-1] == _REFERENCE["last"]
    assert histories == _REFERENCE["warmup_history"]
    assert len(result) == 210
    native = result.attrs["backtest_result"]
    assert native.to_portfolio_state_dataframe().height == 210


@pytest.mark.parametrize("fail", [False, True])
def test_borrowed_runtime_restores_bounds_after_completed_run_or_callback_failure(fail):
    runtime = RuntimeServices()
    seen = []

    def handle(context, data):
        seen.append(context.get_datetime())
        if fail:
            raise RuntimeError("abort bounded run")

    def execute():
        return run_algorithm(
            feed=_feed(),
            assets=[Asset("A")],
            runtime=runtime,
            start="2025-07-03",
            end="2025-07-05",
            handle_data=handle,
        )

    if fail:
        with pytest.raises(RuntimeError, match="abort bounded run"):
            execute()
    else:
        execute()
    assert runtime.start is None and runtime.end is None
    assert seen[0] == pd.Timestamp("2025-07-03T13:31:00Z")
    with pytest.raises(RuntimeError, match="active run"):
        _ = runtime.market


def test_conflicting_native_window_is_rejected_before_initializer_or_runtime_mutation():
    runtime = RuntimeServices(start=datetime(2025, 7, 3, 14, tzinfo=UTC))
    calls = []
    with pytest.raises(ValueError, match="conflict"):
        run_algorithm(
            feed=_feed(),
            assets=[Asset("A")],
            runtime=runtime,
            start="2025-07-03",
            initialize=lambda context: calls.append(context),
        )
    assert calls == [] and runtime.start == datetime(2025, 7, 3, 14, tzinfo=UTC)


@pytest.mark.parametrize(
    "start,end,message",
    [
        ("2025-07-05", "2025-07-03", "start must"),
        ("2025-07-04", "2025-07-06", "no calendar sessions"),
        ("NaT", "2025-07-03", "finite"),
        ("2025-07-03T12:00:00", "2025-07-03", "midnight session labels"),
        ("2025-07-03T00:00:00Z", "2025-07-03", "midnight session labels"),
    ],
)
def test_invalid_session_window_cannot_silently_shift_dates_or_run_initializer(start, end, message):
    calls = []
    with pytest.raises(ValueError, match=message):
        run_algorithm(
            feed=_feed(),
            assets=[Asset("A")],
            start=start,
            end=end,
            initialize=lambda context: calls.append(context),
        )
    assert calls == []


def test_failed_initializer_restores_borrowed_window_for_subsequent_use():
    runtime = RuntimeServices()

    def initialize(context):
        raise RuntimeError("initializer failed")

    with pytest.raises(RuntimeError, match="initializer failed"):
        run_algorithm(
            feed=_feed(),
            assets=[Asset("A")],
            runtime=runtime,
            start="2025-07-03",
            initialize=initialize,
        )
    assert runtime.start is None and runtime.end is None
    result = run_algorithm(feed=_feed(), assets=[Asset("A")], runtime=runtime, start="2025-07-03")
    assert len(result) == 210


def test_calendar_free_run_keeps_physical_window_and_rejects_session_date_interpretation():
    frame = _feed().prices
    feed = DataFeed(
        prices_df=frame,
        feed_spec={
            "calendar": None,
            "timestamp_semantics": "event_time",
            "time_contract_version": 2,
        },
    )
    with pytest.raises(ValueError, match="declared calendar"):
        run_algorithm(feed=feed, assets=[Asset("A")], start="2025-07-03")
    runtime = RuntimeServices(start=datetime(2025, 7, 3, 15, tzinfo=UTC))
    environments = []
    result = run_algorithm(
        feed=feed,
        assets=[Asset("A")],
        runtime=runtime,
        initialize=lambda context: environments.append(context.get_environment("*")),
    )
    assert result.index[0] == pd.Timestamp("2025-07-03T15:00:00Z")
    assert environments[0]["start"] == result.index[0]
    assert runtime.start == datetime(2025, 7, 3, 15, tzinfo=UTC)


def test_session_bounds_localize_naive_feed_instants_without_changing_reference_callbacks():
    original = _feed()
    frame = original.prices.with_columns(
        pl.col("timestamp").dt.convert_time_zone("America/New_York").dt.replace_time_zone(None)
    )
    feed = DataFeed(
        prices_df=frame, feed_spec=original.feed_spec.with_overrides(timezone="America/New_York")
    )
    seen = []
    run_algorithm(
        feed=feed,
        assets=[Asset("A")],
        start="2025-07-03",
        end="2025-07-05",
        handle_data=lambda context, _data: seen.append(context.get_datetime().isoformat()),
    )
    assert len(seen) == 210
    assert seen[0] == _REFERENCE["first"] and seen[-1] == _REFERENCE["last"]


def test_cme_session_label_selects_previous_evening_without_confusing_utc_dates():
    times = [
        datetime(2025, 7, 6, 23, tzinfo=UTC),
        datetime(2025, 7, 7, 0, tzinfo=UTC),
        datetime(2025, 7, 7, 21, tzinfo=UTC),
    ]
    feed = DataFeed(
        prices_df=pl.DataFrame({"timestamp": times, "asset": ["A"] * 3, "close": [100.0] * 3}),
        feed_spec={
            "calendar": "CME_Equity",
            "timestamp_semantics": "bar_close",
            "time_contract_version": 2,
            "data_frequency": "1m",
            "bar_label": "right",
            "bar_grid_anchor": "session_open",
            "partial_bar_policy": "include",
        },
    )
    environments = []
    result = run_algorithm(
        feed=feed,
        assets=[Asset("A")],
        start="2025-07-07",
        end="2025-07-07",
        initialize=lambda context: environments.append(context.get_environment("*")),
    )
    assert result.index.tolist() == [pd.Timestamp(time) for time in times]
    assert environments[0]["start"] == pd.Timestamp("2025-07-06T22:01:00Z")
    assert environments[0]["end"] == pd.Timestamp("2025-07-07T21:00:00Z")
